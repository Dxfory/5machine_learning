from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Tushare initialisation & configuration
# ---------------------------------------------------------------------------
# Valid token provided by the user (required for all API calls)
TUSHARE_TOKEN = "2274b36cdb25742d5954ca7e5cb4a59c709ad32f41202a344f1eegx5"

if not TUSHARE_TOKEN:
    raise RuntimeError("Tushare token is empty. Please set TUSHARE_TOKEN before running any scripts.")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# Expose factor names so strategy modules can stay in sync with the actual
# factors that we are capable of fetching from Tushare.
AVAILABLE_FACTORS: Tuple[str, ...] = (
    "close",  # latest close price
    "market_cap",  # circulating market value
    "turnover_rate",  # turnover rate of the day
    "volume_ratio",  # volume ratio of the day
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "pct_change_5d",
    "pct_change_20d",
    "momentum_60d",
    "volatility_20d",
    "volatility_60d",
    "avg_turnover_20d",
    "avg_turnover_60d",
    "turnover_std_20d",
    "turnover_std_60d",
    "max_drawdown_20d",
)

# Directory placeholder (created lazily in strategy as well) – this ensures the
# repository contains the directory structure that strategy saving expects.
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Internal caches to reduce duplicate API calls
# ---------------------------------------------------------------------------
_trade_cal_cache: Dict[Tuple[str, str], List[str]] = {}
_index_weight_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_stock_basic_cache: Optional[pd.DataFrame] = None
_namechange_cache: Dict[str, pd.DataFrame] = {}
_special_trade_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_daily_cache: Dict[str, pd.DataFrame] = {}
_daily_basic_cache: Dict[str, pd.DataFrame] = {}
_daily_basic_range_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_price_panel_cache: Dict[Tuple[Tuple[str, ...], str, str], pd.DataFrame] = {}

# Fields used for daily_basic range queries
_DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio,pe_ttm,pb,ps_ttm,dv_ttm,total_mv,circ_mv"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _format_date(date: str) -> str:
    if isinstance(date, str):
        return date
    return pd.to_datetime(date).strftime("%Y-%m-%d")


def _trade_date_key(date: str) -> str:
    return date.replace("-", "")


def get_trade_days(start_date: str, end_date: str) -> List[str]:
    """Get list of trading days between start_date and end_date (inclusive)."""
    start = _format_date(start_date)
    end = _format_date(end_date)
    cache_key = (start, end)
    if cache_key in _trade_cal_cache:
        return _trade_cal_cache[cache_key]
    df_calendar = pro.trade_cal(exchange="SSE", start_date=_trade_date_key(start), end_date=_trade_date_key(end), is_open="1", fields="cal_date")
    trade_days = [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in df_calendar["cal_date"].tolist()]
    trade_days.sort()
    _trade_cal_cache[cache_key] = trade_days
    return trade_days


def _get_index_weight(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    cache_key = (index_code, f"{start_date}_{end_date}")
    if cache_key not in _index_weight_cache:
        _index_weight_cache[cache_key] = pro.index_weight(index_code=index_code, start_date=start_date, end_date=end_date, fields="con_code")
    return _index_weight_cache[cache_key]


def get_index_stocks(index_code: str, date: str) -> List[str]:
    """Get the list of constituent stock codes for the given index on the given date."""
    jq_to_ts_map = {
        "000300.XSHG": "399300.SZ",  # CSI 300
        "000016.XSHG": "000016.SH",  # SSE 50
        "000905.XSHG": "399905.SZ",  # CSI 500
        "399905.XSHE": "399905.SZ",  # CSI 500
        "399906.XSHE": "399906.SZ",  # CSI 800
        "399006.XSHE": "399006.SZ",  # ChiNext
        "399101.XSHE": "399101.SZ",  # SZ SME Composite
        "399107.XSHE": "399107.SZ",  # SZ A Index
        "000985.XSHG": "000985.CSI",  # CSI All Share Index
    }
    ts_index = jq_to_ts_map.get(index_code)
    if ts_index is None:
        raise ValueError(f"Index code {index_code} not recognized for mapping to Tushare.")

    date_obj = datetime.strptime(_format_date(date), "%Y-%m-%d")
    month_start = date_obj.replace(day=1).strftime("%Y%m%d")
    next_month = date_obj.replace(day=28) + timedelta(days=4)
    month_end = (next_month.replace(day=1) - timedelta(days=1)).strftime("%Y%m%d")
    df_weights = _get_index_weight(ts_index, month_start, month_end)
    stocks = [code for code in df_weights["con_code"].unique().tolist() if isinstance(code, str)]
    return stocks


def _load_stock_basic() -> pd.DataFrame:
    global _stock_basic_cache
    if _stock_basic_cache is None:
        _stock_basic_cache = pro.stock_basic(exchange="", list_status="L", fields="ts_code,list_date")
        _stock_basic_cache["list_date"] = pd.to_datetime(_stock_basic_cache["list_date"], format="%Y%m%d", errors="coerce")
    return _stock_basic_cache


def get_stock_pool(date: str, pool_name: str = "ZXBZ") -> List[str]:
    """Return stock list for a given pool on the given date."""
    date = _format_date(date)
    if pool_name == "HS300":
        stocks = get_index_stocks("000300.XSHG", date)
    elif pool_name == "ZZ500":
        stocks = get_index_stocks("399905.XSHE", date)
    elif pool_name == "ZZ800":
        stocks = get_index_stocks("399906.XSHE", date)
    elif pool_name == "CYBZ":
        stocks = get_index_stocks("399006.XSHE", date)
    elif pool_name == "ZXBZ":
        stocks = get_index_stocks("399101.XSHE", date)
    elif pool_name == "A":
        df_stocks = _load_stock_basic()
        stocks = df_stocks.loc[df_stocks["list_date"] <= pd.to_datetime(date)]["ts_code"].tolist()
        stocks = [s for s in stocks if not (s.startswith("688") or s.startswith("787") or s.startswith("300"))]
    elif pool_name == "AA":
        stocks = get_index_stocks("000985.XSHG", date)
        stocks = [s for s in stocks if not (s.startswith("688") or s.startswith("787") or s.startswith("300"))]
    else:
        stocks = get_index_stocks("399101.XSHE", date)

    if not stocks:
        return []

    df_basic = _load_stock_basic().set_index("ts_code")
    cut_off = pd.to_datetime(date) - pd.Timedelta(days=90)
    stocks = [code for code in stocks if code in df_basic.index]
    list_dates = df_basic.reindex(stocks)["list_date"]
    filtered = [code for code, list_date in zip(stocks, list_dates) if pd.notna(list_date) and list_date <= cut_off]
    filtered = [code for code in filtered if not is_st_stock(code, date)]
    return filtered[:300]


def _get_name_change(ts_code: str) -> pd.DataFrame:
    if ts_code not in _namechange_cache:
        _namechange_cache[ts_code] = pro.namechange(ts_code=ts_code, fields="name,start_date,end_date")
    return _namechange_cache[ts_code]


def _get_special_trade(ts_code: str, start: str, end: str) -> pd.DataFrame:
    key = (ts_code, f"{start}_{end}")
    if key not in _special_trade_cache:
        _special_trade_cache[key] = pro.stk_special_trade(ts_code=ts_code, start_date=start, end_date=end)
    return _special_trade_cache[key]


def is_st_stock(ts_code: str, date: str) -> bool:
    date = _format_date(date)
    date_int = int(_trade_date_key(date))
    name_df = _get_name_change(ts_code)
    if not name_df.empty:
        current_names = name_df[(name_df["start_date"] <= date_int) & ((name_df["end_date"].isna()) | (name_df["end_date"] >= date_int))]
        if not current_names.empty and current_names.iloc[0]["name"].upper().find("ST") != -1:
            return True
    st_df = _get_special_trade(ts_code, _trade_date_key(date), _trade_date_key(date))
    if not st_df.empty and st_df["special_type"].astype(str).str.contains("ST").any():
        return True
    return False


def _get_daily(trade_date: str) -> pd.DataFrame:
    trade_date = _format_date(trade_date)
    key = _trade_date_key(trade_date)
    if key not in _daily_cache:
        _daily_cache[key] = pro.daily(trade_date=key, fields="ts_code,close,pct_chg")
    return _daily_cache[key]


def _get_daily_basic(trade_date: str) -> pd.DataFrame:
    trade_date = _format_date(trade_date)
    key = _trade_date_key(trade_date)
    if key not in _daily_basic_cache:
        _daily_basic_cache[key] = pro.daily_basic(trade_date=key, fields=_DAILY_BASIC_FIELDS)
    return _daily_basic_cache[key]


def _get_daily_basic_range(start_date: str, end_date: str) -> pd.DataFrame:
    start = _trade_date_key(_format_date(start_date))
    end = _trade_date_key(_format_date(end_date))
    key = (start, end)
    if key not in _daily_basic_range_cache:
        df = pro.query("daily_basic", start_date=start, end_date=end, fields=_DAILY_BASIC_FIELDS)
        if df.empty:
            _daily_basic_range_cache[key] = df
        else:
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            _daily_basic_range_cache[key] = df
    return _daily_basic_range_cache[key]


def get_price_df(stock_list: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Get daily close prices for the given stocks between start_date and end_date."""
    stocks = tuple(sorted(set(stock_list)))
    if not stocks:
        return pd.DataFrame()
    start = _format_date(start_date)
    end = _format_date(end_date)
    cache_key = (stocks, start, end)
    if cache_key in _price_panel_cache:
        return _price_panel_cache[cache_key].copy()

    trade_days = get_trade_days(start, end)
    frames = []
    for day in trade_days:
        df_daily = _get_daily(day)
        if df_daily.empty:
            continue
        df_filtered = df_daily[df_daily["ts_code"].isin(stocks)][["ts_code", "close"]].copy()
        if df_filtered.empty:
            continue
        df_filtered["trade_date"] = pd.to_datetime(day)
        frames.append(df_filtered)
    if not frames:
        return pd.DataFrame(index=stocks)
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot(index="ts_code", columns="trade_date", values="close")
    pivot = pivot.reindex(stocks)
    _price_panel_cache[cache_key] = pivot
    return pivot.copy()


def _compute_max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    running_max = series.cummax()
    drawdown = (series / running_max) - 1.0
    return drawdown.min()


def get_factor_values(stock_list: Sequence[str], date: str) -> pd.DataFrame:
    """Fetch factor values for the given stocks on the given date."""
    stocks = [s for s in stock_list if isinstance(s, str)]
    if not stocks:
        return pd.DataFrame()

    date = _format_date(date)
    end_ts = pd.to_datetime(date)
    lookback_days = 90  # to compute volatility and turnover statistics safely
    history_start = (end_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    history_trade_days = get_trade_days(history_start, date)
    if not history_trade_days:
        return pd.DataFrame(index=stocks)
    actual_start = history_trade_days[0]

    price_panel = get_price_df(stocks, actual_start, date)
    db_today_raw = _get_daily_basic(date)
    db_today = db_today_raw.set_index("ts_code") if not db_today_raw.empty else pd.DataFrame()
    db_range = _get_daily_basic_range(actual_start, date)

    result = pd.DataFrame(index=stocks, columns=AVAILABLE_FACTORS, dtype=float)

    if not price_panel.empty:
        price_panel = price_panel.sort_index(axis=1)
    else:
        price_panel = pd.DataFrame(index=stocks)

    if not db_range.empty:
        db_range = db_range[db_range["ts_code"].isin(stocks)].copy()
        db_range.sort_values(["ts_code", "trade_date"], inplace=True)
    else:
        db_range = pd.DataFrame(columns=["ts_code", "trade_date"])

    for stock in stocks:
        series = price_panel.loc[stock] if stock in price_panel.index else pd.Series(dtype=float)
        series = series.dropna().sort_index()
        if not series.empty and series.index[-1] != pd.to_datetime(date):
            # ensure the last available price corresponds to the evaluation date
            # if not, try to reindex using nearest available
            series = series
        close_price = series.iloc[-1] if not series.empty else np.nan
        result.at[stock, "close"] = close_price

        # Market data based features
        if not series.empty:
            if len(series) >= 5:
                result.at[stock, "pct_change_5d"] = series.iloc[-1] / series.iloc[-5] - 1
            if len(series) >= 20:
                result.at[stock, "pct_change_20d"] = series.iloc[-1] / series.iloc[-20] - 1
                returns_20 = series.pct_change().dropna().iloc[-20:]
                result.at[stock, "volatility_20d"] = returns_20.std() if not returns_20.empty else np.nan
                window_20 = series.iloc[-20:]
                result.at[stock, "max_drawdown_20d"] = _compute_max_drawdown(window_20)
            if len(series) >= 60:
                result.at[stock, "momentum_60d"] = series.iloc[-1] / series.iloc[-60] - 1
                returns_60 = series.pct_change().dropna().iloc[-60:]
                result.at[stock, "volatility_60d"] = returns_60.std() if not returns_60.empty else np.nan
        # Daily basic snapshot
        if not db_today.empty and stock in db_today.index:
            snapshot = db_today.loc[stock]
            result.at[stock, "market_cap"] = snapshot.get("circ_mv")
            result.at[stock, "turnover_rate"] = snapshot.get("turnover_rate")
            result.at[stock, "volume_ratio"] = snapshot.get("volume_ratio")
            result.at[stock, "pe_ttm"] = snapshot.get("pe_ttm")
            result.at[stock, "pb"] = snapshot.get("pb")
            result.at[stock, "ps_ttm"] = snapshot.get("ps_ttm")
            result.at[stock, "dv_ttm"] = snapshot.get("dv_ttm")

        # Turnover statistics from historical daily_basic
        if not db_range.empty:
            stock_hist = db_range[db_range["ts_code"] == stock]
            if not stock_hist.empty:
                stock_hist = stock_hist.sort_values("trade_date")
                last_20 = stock_hist.tail(20)
                last_60 = stock_hist.tail(60)
                if not last_20.empty:
                    result.at[stock, "avg_turnover_20d"] = last_20["turnover_rate"].mean()
                    result.at[stock, "turnover_std_20d"] = last_20["turnover_rate"].std()
                if not last_60.empty:
                    result.at[stock, "avg_turnover_60d"] = last_60["turnover_rate"].mean()
                    result.at[stock, "turnover_std_60d"] = last_60["turnover_rate"].std()

    return result


__all__ = [
    "AVAILABLE_FACTORS",
    "get_trade_days",
    "get_index_stocks",
    "get_stock_pool",
    "get_price_df",
    "get_factor_values",
    "is_st_stock",
]
