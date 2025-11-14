import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Tushare pro API
ts.set_token('YOUR_TUSHARE_TOKEN')
pro = ts.pro_api()

def get_trade_days(start_date, end_date):
    """Get list of trading days between start_date and end_date (inclusive)."""
    start = start_date.replace('-', '')
    end = end_date.replace('-', '')
    df_calendar = pro.trade_cal(exchange='SSE', start_date=start, end_date=end, is_open='1',
                                fields='cal_date')  # SSE calendar, is_open=1 for trading days
    trade_days = df_calendar['cal_date'].tolist()
    # Convert to YYYY-MM-DD format
    trade_days = [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in trade_days]
    trade_days.sort()
    return trade_days

def get_index_stocks(index_code, date):
    """Get the list of constituent stock codes for the given index on the given date."""
    # Map JoinQuant style index codes to Tushare index_code
    jq_to_ts_map = {
        # Major indices
        '000300.XSHG': '399300.SZ',  # CSI 300
        '000016.XSHG': '000016.SH',  # SSE 50
        '000905.XSHG': '399905.SZ',  # CSI 500 (if used)
        '399905.XSHE': '399905.SZ',  # CSI 500
        '399906.XSHE': '399906.SZ',  # CSI 800
        '399006.XSHE': '399006.SZ',  # ChiNext Index
        '399101.XSHE': '399101.SZ',  # SZSE SME Composite (assumed for ZXBZ)
        '399107.XSHE': '399107.SZ',  # SZ A Index (for all SZ A shares)
        '000985.XSHG': '000985.CSI', # CSI All Share Index (AA) – using CSI code if available
        # ... add other mappings as needed
    }
    ts_index = jq_to_ts_map.get(index_code, None)
    if ts_index is None:
        raise ValueError(f"Index code {index_code} not recognized for mapping to Tushare.")
    # Tushare index_weight is monthly; get that month’s composition
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    month_start = date_obj.replace(day=1).strftime("%Y%m%d")
    # Set end_date as last day of month
    next_month = date_obj.replace(day=28) + timedelta(days=4)  # this will definitely be next month
    month_end = next_month.replace(day=1) - timedelta(days=1)
    month_end = month_end.strftime("%Y%m%d")
    df_weights = pro.index_weight(index_code=ts_index, start_date=month_start, end_date=month_end,
                                  fields="con_code")
    stocks = df_weights['con_code'].unique().tolist()
    # Filter out any empty or None codes
    stocks = [code for code in stocks if isinstance(code, str)]
    # Format codes to match expected style (Tushare codes are like 600000.SH)
    return stocks

def get_stock_pool(date, pool_name='ZXBZ'):
    """Return stock list for a given pool on the given date (e.g., HS300, A, AA, etc.)."""
    date_str = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    if pool_name == 'HS300':
        stocks = get_index_stocks('000300.XSHG', date)
    elif pool_name == 'ZZ500':
        stocks = get_index_stocks('399905.XSHE', date)
    elif pool_name == 'ZZ800':
        stocks = get_index_stocks('399906.XSHE', date)
    elif pool_name == 'CYBZ':
        stocks = get_index_stocks('399006.XSHE', date)
    elif pool_name == 'ZXBZ':
        stocks = get_index_stocks('399101.XSHE', date)
    elif pool_name == 'A':
        # All A shares (exclude STAR and ChiNext)
        df_stocks = pro.stock_basic(exchange='', list_status='L',
                                    fields='ts_code, list_date')
        # Filter by date: include those listed at least by the given date
        df_stocks = df_stocks[df_stocks['list_date'] <= date_str]
        stocks = df_stocks['ts_code'].tolist()
        # Exclude Sci-Tech board (STAR) and ChiNext codes by prefix
        stocks = [s for s in stocks if not (s.startswith('688') or s.startswith('787')
                                            or s.startswith('300'))]
    elif pool_name == 'AA':
        # CSI All A index, then filter out STAR/ChiNext similar to 'A'
        stocks = get_index_stocks('000985.XSHG', date)
        stocks = [s for s in stocks if not (s.startswith('688') or s.startswith('787')
                                            or s.startswith('300'))]
    else:
        # Default to ZXBZ (Middle/Small Board Composite)
        stocks = get_index_stocks('399101.XSHE', date)
    # Filter ST stocks (remove stocks that are ST on this date)
    stocks = [code for code in stocks if not is_st_stock(code, date)]
    # Filter newly listed stocks (listed within 90 days before date)
    filtered = []
    for code in stocks:
        info = pro.stock_basic(ts_code=code, fields='ts_code,list_date')
        if info.empty:
            continue
        list_date = datetime.strptime(info.iloc[0]['list_date'], "%Y%m%d").date()
        if (datetime.strptime(date, "%Y-%m-%d").date() - list_date).days > 90:
            filtered.append(code)
    return filtered[:300]  # limit to 300 if needed (as in original)

def is_st_stock(ts_code, date):
    """Determine if the stock was ST on the given date by checking its name or special trade info."""
    # Approach 1: Check name for "ST"
    name_df = pro.namechange(ts_code=ts_code, fields='name,start_date,end_date')
    # namechange returns all historical names with date ranges
    if not name_df.empty:
        date_int = int(date.replace('-', ''))
        current_names = name_df[(name_df['start_date'] <= date_int) &
                                 ((name_df['end_date'].isna()) | (name_df['end_date'] >= date_int))]
        if not current_names.empty:
            name = current_names.iloc[0]['name']
            if 'ST' in name:
                return True
    # Approach 2: Use special trade (ST) list
    st_df = pro.stk_special_trade(ts_code=ts_code, start_date=date.replace('-',''), end_date=date.replace('-',''))
    # This returns records if the stock had special trade (like ST/*ST) on that date
    if not st_df.empty:
        # If there's an entry with `special_type` indicating ST status
        if any(st_df['special_type'].str.contains('ST')):
            return True
    return False

def get_price_df(stock_list, start_date, end_date):
    """Get daily close prices for the given stocks between start_date and end_date."""
    start = start_date.replace('-', '')
    end = end_date.replace('-', '')
    all_prices = {}
    # Retrieve all daily data day by day for efficiency (since pro.daily allows filtering by date)
    trade_days = get_trade_days(start_date, end_date)
    for day in trade_days:
        day_str = day.replace('-', '')
        df = pro.daily(trade_date=day_str, fields='ts_code, close')
        if df.empty:
            continue
        # Filter to the stock_list for performance
        df = df[df['ts_code'].isin(stock_list)]
        for _, row in df.iterrows():
            code = row['ts_code']
            price = row['close']
            if code not in all_prices:
                all_prices[code] = []
            all_prices[code].append((day, price))
    # Convert to DataFrame for convenience: index = stock, columns = date, values = close
    price_df = pd.DataFrame({code: {pd.to_datetime(day): price for day, price in data}
                              for code, data in all_prices.items()}).T
    # Sort the price series by date (columns sorted)
    price_df = price_df.sort_index(axis=1)
    return price_df

def get_factor_values(stock_list, date):
    """Fetch all required factor values for the given stocks on the given date."""
    date_str = date.replace('-', '')
    factors = {}
    # Get technical factors from Tushare (including price and volume data for that date)
    df_factor = pro.stk_factor_pro(trade_date=date_str)
    if not df_factor.empty:
        df_factor = df_factor.set_index('ts_code')
    # For each factor in our factor list, extract or compute:
    # Technical factors directly from df_factor:
    tech_map = {
        # map our factor name to the corresponding column in df_factor (bfq for unadjusted by default)
        'AR': 'brar_ar_bfq',
        'ARBR': 'brar_br_bfq',  # BR
        'ATR6': None,   # will compute separately
        'DAVOL10': None, # compute separately (10-day avg volume or ratio)
        'MAWVAD': None,  # compute WVAD indicator
        'PSY': 'psy_bfq',  # 12-day Psychological line
        'VOL10': None, 'VOL120': None,  # average volumes or volatility? assume avg volume last 10/120 days
        'VDIFF': None, 'VEMA26': None, 'VMACD': None, 'VOSC': None, # compute volume MACD
        'VR': 'vr_bfq',
        'WVAD': None,  # compute WVAD
        'arron_down_25': None, 'arron_up_25': None,  # compute Aroon
        'BBIC': 'bbi_bfq',  # Bull&Bear Index (BBI)
        'MASS': 'mass_bfq', # Mass index
        'Rank1M': None,     # 1-month return rank or similar
        'single_day_VPT': None, 'single_day_VPT_12': None, 'single_day_VPT_6': None,  # compute VPT
        'Volume1M': None,   # volume last 1 month
        'MFI14': 'mfi_bfq',
        'boll_down': 'boll_lower_bfq',  # lower Bollinger band
        'MAC10': None,  # Possibly 10-day moving average convergence? (Not clear, assume 10-day MACD signal)
        'fifty_two_week_close_rank': None,  # compute 52-week rank
        'price_no_fq': None  # unadjusted price = close_bfq which is just 'close' field
    }
    # Direct technical factors from df_factor:
    for our_factor, column in tech_map.items():
        if column and our_factor in tech_map:
            # If factor is directly available
            try:
                factors[our_factor] = df_factor.loc[stock_list, column]
            except KeyError:
                # If some stock missing in factor DataFrame, fill with NaN
                values = []
                for stock in stock_list:
                    values.append(df_factor.at[stock, column] if stock in df_factor.index else np.nan)
                factors[our_factor] = pd.Series(values, index=stock_list)
    # Compute missing technicals using historical data if needed:
    # For efficiency, gather required history only once:
    needed_history_days = {}
    if 'ATR6' in tech_map or 'arron_down_25' in tech_map:
        needed_history_days['ATR6'] = 6
    if 'arron_down_25' in tech_map or 'arron_up_25' in tech_map:
        needed_history_days['Aroon25'] = 25
    if 'VOL10' in tech_map or 'DAVOL10' in tech_map:
        needed_history_days['VOL10'] = 10
    if 'VOL120' in tech_map:
        needed_history_days['VOL120'] = 120
    if 'single_day_VPT' in tech_map or 'single_day_VPT_6' in tech_map or 'single_day_VPT_12' in tech_map:
        needed_history_days['VPT12'] = 12
    if 'fifty_two_week_close_rank' in tech_map or 'momentum' in tech_map:
        needed_history_days['FiftyTwoWeek'] = 252
    # Fetch price and volume history for the max window needed (e.g., max of needed_history_days values)
    if needed_history_days:
        max_days = max(needed_history_days.values())
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = (end_date - timedelta(days=max_days*2))  # multiply by 2 to account for weekends/holidays roughly
        price_data = get_price_df(stock_list, start_date.strftime("%Y-%m-%d"), date)
        # price_data: DataFrame indexed by stock, columns = dates, values = close
        # Also need High, Low, Volume for ATR and others:
        # Fetch OHLCV for needed stocks and needed period
        ohlcv_dict = {}
        for stock in stock_list:
            df_hist = pro.daily(ts_code=stock, start_date=start_date.strftime("%Y%m%d"), end_date=date_str,
                                 fields='trade_date, open, close, high, low, vol')
            df_hist.sort_values('trade_date', inplace=True)
            ohlcv_dict[stock] = df_hist  # store full DataFrame for indicator calcs
    # Now compute specific factors:
    if 'ATR6' in tech_map:
        atr_values = {}
        for stock in stock_list:
            if stock in ohlcv_dict:
                df = ohlcv_dict[stock]
                # Compute True Range and ATR
                highs = df['high'].astype(float)
                lows = df['low'].astype(float)
                closes = df['close'].astype(float)
                # True range = max(high, prev_close) - min(low, prev_close)
                prev_close = closes.shift(1)
                tr = np.maximum(highs, prev_close) - np.minimum(lows, prev_close)
                atr6 = tr.rolling(window=6).mean().iloc[-1]
                atr_values[stock] = atr6
            else:
                atr_values[stock] = np.nan
        factors['ATR6'] = pd.Series(atr_values)
    if 'arron_down_25' in tech_map or 'arron_up_25' in tech_map:
        aroon_down_vals = {}
        aroon_up_vals = {}
        period = 25
        for stock in stock_list:
            if stock in ohlcv_dict:
                df = ohlcv_dict[stock]
                lows = df['low'].astype(float)
                highs = df['high'].astype(float)
                if len(df) >= period:
                    recent_highs = highs.iloc[-period:]
                    recent_lows = lows.iloc[-period:]
                    # Days since last peak/trough:
                    days_since_high = period - 1 - recent_highs.argmax()
                    days_since_low = period - 1 - recent_lows.argmin()
                    aroon_up = ((period - days_since_high) / period) * 100
                    aroon_down = ((period - days_since_low) / period) * 100
                else:
                    aroon_up = np.nan
                    aroon_down = np.nan
            else:
                aroon_up = np.nan
                aroon_down = np.nan
            aroon_down_vals[stock] = aroon_down
            aroon_up_vals[stock] = aroon_up
        factors['arron_down_25'] = pd.Series(aroon_down_vals)
        factors['arron_up_25'] = pd.Series(aroon_up_vals)
    # Volume-based factors (VPT, VOSC, VDIFF, VMACD, etc.)
    if 'single_day_VPT' in tech_map or 'single_day_VPT_6' in tech_map or 'single_day_VPT_12' in tech_map:
        vpt_vals = {}
        vpt6_vals = {}
        vpt12_vals = {}
        for stock in stock_list:
            if stock in ohlcv_dict:
                df = ohlcv_dict[stock]
                closes = df['close'].astype(float)
                vols = df['vol'].astype(float)
                # Compute cumulative VPT: VPT[t] = VPT[t-1] + volume[t] * (close[t] - close[t-1]) / close[t-1]
                vpt = 0.0
                vpt_series = []
                prev_close_val = None
                for c, v in zip(closes, vols):
                    if prev_close_val is not None and prev_close_val != 0:
                        vpt += v * (c - prev_close_val) / prev_close_val
                    vpt_series.append(vpt)
                    prev_close_val = c
                # Take last day's VPT values for required periods
                vpt_vals[stock] = vpt_series[-1] if vpt_series else np.nan
                if len(vpt_series) >= 6:
                    vpt6_vals[stock] = vpt_series[-1] - vpt_series[-7]  # single day VPT change over 6 days?
                else:
                    vpt6_vals[stock] = np.nan
                if len(vpt_series) >= 12:
                    vpt12_vals[stock] = vpt_series[-1] - vpt_series[-13]
                else:
                    vpt12_vals[stock] = np.nan
            else:
                vpt_vals[stock] = np.nan
                vpt6_vals[stock] = np.nan
                vpt12_vals[stock] = np.nan
        factors['single_day_VPT'] = pd.Series(vpt_vals)
        factors['single_day_VPT_6'] = pd.Series(vpt6_vals)
        factors['single_day_VPT_12'] = pd.Series(vpt12_vals)
    if 'VDIFF' in tech_map or 'VEMA26' in tech_map or 'VMACD' in tech_map or 'VOSC' in tech_map:
        vdiff_vals = {}
        vema26_vals = {}
        vmacd_vals = {}
        vosc_vals = {}
        for stock in stock_list:
            if stock in ohlcv_dict:
                vols = ohlcv_dict[stock]['vol'].astype(float)
                # compute EMA12 and EMA26 of volume
                ema12 = vols.ewm(span=12, adjust=False).mean()
                ema26 = vols.ewm(span=26, adjust=False).mean()
                if not ema26.empty:
                    diff = ema12.iloc[-1] - ema26.iloc[-1]
                    signal = diff  # default if no previous, we'll compute below
                    # EMA of diff with span=9 for signal
                    diff_series = ema12 - ema26
                    signal_series = diff_series.ewm(span=9, adjust=False).mean()
                    signal = signal_series.iloc[-1]
                    vdiff_vals[stock] = diff
                    vema26_vals[stock] = ema26.iloc[-1]
                    vmacd_vals[stock] = diff - signal  # MACD histogram
                    if ema26.iloc[-1] != 0:
                        vosc_vals[stock] = (ema12.iloc[-1] - ema26.iloc[-1]) / ema26.iloc[-1] * 100
                    else:
                        vosc_vals[stock] = np.nan
                else:
                    vdiff_vals[stock] = np.nan
                    vema26_vals[stock] = np.nan
                    vmacd_vals[stock] = np.nan
                    vosc_vals[stock] = np.nan
            else:
                vdiff_vals[stock] = np.nan
                vema26_vals[stock] = np.nan
                vmacd_vals[stock] = np.nan
                vosc_vals[stock] = np.nan
        factors['VDIFF'] = pd.Series(vdiff_vals)
        factors['VEMA26'] = pd.Series(vema26_vals)
        factors['VMACD'] = pd.Series(vmacd_vals)
        factors['VOSC'] = pd.Series(vosc_vals)
    if 'WVAD' in tech_map:
        wvad_vals = {}
        period = 24  # typically 24-day period for Williams VAD
        for stock in stock_list:
            if stock in ohlcv_dict:
                df = ohlcv_dict[stock]
                if len(df) >= period:
                    recent = df.iloc[-period:]
                    # WVAD = sum((Close - Open)/(High - Low) * Volume) over period
                    num = (recent['close'] - recent['open']) / (recent['high'] - recent['low'])
                    wvad = (num * recent['vol']).sum()
                else:
                    wvad = np.nan
            else:
                wvad = np.nan
            wvad_vals[stock] = wvad
        factors['WVAD'] = pd.Series(wvad_vals)
    # 1-month rank (Rank1M): rank of each stock's 1-month return among all stocks
    if 'Rank1M' in tech_map:
        one_month_returns = {}
        # define 1 month ~ 21 trading days
        window = 21
        for stock in stock_list:
            if stock in price_data.index and price_data.shape[1] > window:
                prices = price_data.loc[stock].dropna()
                if len(prices) >= window:
                    ret = prices.iloc[-1] / prices.iloc[-window] - 1
                else:
                    ret = np.nan
            else:
                ret = np.nan
            one_month_returns[stock] = ret
        # Rank the returns
        ser = pd.Series(one_month_returns)
        ranks = ser.rank(pct=True)
        factors['Rank1M'] = ranks
    # Volume1M: total volume in last 21 days
    if 'Volume1M' in tech_map:
        vol1m_vals = {}
        for stock in stock_list:
            if stock in ohlcv_dict:
                df = ohlcv_dict[stock]
                vol = df['vol'].astype(float)
                if len(vol) >= 21:
                    vol1m_vals[stock] = vol.iloc[-21:].sum()
                else:
                    vol1m_vals[stock] = vol.sum()
            else:
                vol1m_vals[stock] = np.nan
        factors['Volume1M'] = pd.Series(vol1m_vals)
    # VOL10, VOL120: average volume over last 10 and 120 trading days
    if 'VOL10' in tech_map or 'VOL120' in tech_map:
        vol10_vals = {}
        vol120_vals = {}
        for stock in stock_list:
            if stock in ohlcv_dict:
                vol = ohlcv_dict[stock]['vol'].astype(float)
                vol10_vals[stock] = vol.iloc[-10:].mean() if len(vol) >= 10 else np.nan
                vol120_vals[stock] = vol.iloc[-120:].mean() if len(vol) >= 120 else np.nan
            else:
                vol10_vals[stock] = np.nan
                vol120_vals[stock] = np.nan
        if 'VOL10' in tech_map:
            factors['VOL10'] = pd.Series(vol10_vals)
        if 'VOL120' in tech_map:
            factors['VOL120'] = pd.Series(vol120_vals)
    # Fundamental factors:
    fundamental_fields = [
        'asset_impairment_loss', 'gross_profit', 'non_operating_profit',
        'operating_profit', 'ebit', 'ebitda', 'total_profit', 'net_profit',
        'financial_assets', 'np_parent_company_q',  # add any needed fields
        # Ratios directly available from fina_indicator:
        'roe', 'roa', 'debt_to_assets', 'debt_to_equity', 'current_ratio', 'quick_ratio',
        'ocf_to_profit', 'ocf_to_liabilities'
    ]
    # We will fetch the latest financial indicator for each stock (last reported quarter)
    fin_df_list = []
    latest_period = None
    # Determine which quarter to use based on date (avoid future data):
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    year = date_dt.year
    quarter = (date_dt.month - 1)//3 + 1  # quarter of the year
    if quarter == 1:
        # Use last year's Q3 as latest available (most Q4 not out yet by Q1)
        latest_period = f"{year-1}Q3"
    elif quarter == 2:
        # Q4 of last year should be available by Q2
        latest_period = f"{year-1}Q4"
    elif quarter == 3:
        # Use current year's Q1
        latest_period = f"{year}Q1"
    else:  # quarter 4
        # Use current year's Q2 (Q3 reports may not all be out until end of Oct)
        latest_period = f"{year}Q2"
    for stock in stock_list:
        fin = pro.fina_indicator(ts_code=stock, period=latest_period)
        if not fin.empty:
            fin.set_index('ts_code', inplace=True)
            fin_df_list.append(fin)
        else:
            # If no data for this period (maybe stock IPO after or missing), try previous period
            alt_period = f"{year-1}Q4"
            fin = pro.fina_indicator(ts_code=stock, period=alt_period)
            if not fin.empty:
                fin.set_index('ts_code', inplace=True)
                fin_df_list.append(fin)
    if fin_df_list:
        fin_df = pd.concat(fin_df_list)
    else:
        fin_df = pd.DataFrame()
    # Compute or extract fundamental factor values:
    fund_factors = {}
    # Example calculations:
    for stock in stock_list:
        if stock in fin_df.index:
            row = fin_df.loc[stock]
            # asset_impairment_loss_ttm: use ratio if available or approximate via income statement
            if 'asset_impairment_loss_ttm' in fund_factors or 'asset_impairment_loss' in row:
                # if we had an indicator impai_ttm (impairment loss/revenue), could multiply by revenue
                fund_factors.setdefault('asset_impairment_loss_ttm', {})[stock] = row.get('impai_ttm', np.nan)
            # cash_flow_to_price_ratio: operating cash flow per share / price
            if 'ocfps' in row and stock in df_factor.index:
                # ocfps = operating cash flow per share (元)
                price = df_factor.at[stock, 'close'] if stock in df_factor.index else np.nan
                fund_factors.setdefault('cash_flow_to_price_ratio', {})[stock] = (row['ocfps']/price) if price and price != 0 else np.nan
            # market_cap: use total_mv from daily_basic (already in df_factor as total_mv in万元)
            if stock in df_factor.index:
                mv = df_factor.at[stock, 'total_mv']  # in thousand RMB
                fund_factors.setdefault('market_cap', {})[stock] = mv * 1000  # convert to RMB
            # interest_free_current_liability: not directly available, compute from balance sheet if needed
            # ... (similar pattern for each required factor)
            # net_asset_per_share: can use bps (book value per share) from fina_indicator
            if 'bps' in row:
                fund_factors.setdefault('net_asset_per_share', {})[stock] = row['bps']
            # total_operating_revenue_per_share: use revenue_ps if available
            if 'revenue_ps' in row:
                fund_factors.setdefault('total_operating_revenue_per_share', {})[stock] = row['revenue_ps']
            # net_operate_cash_flow_per_share: use ocfps
            if 'ocfps' in row:
                fund_factors.setdefault('net_operate_cash_flow_per_share', {})[stock] = row['ocfps']
            # operating_profit_per_share: can compute as operating_profit/total_shares
            if 'opincome' in row and 'total_share' in df_factor.columns:
                shares = df_factor.at[stock, 'total_share'] if stock in df_factor.index else None
                if shares:
                    op_profit = row['opincome']  # operating profit (单季度 or TTM?)
                    fund_factors.setdefault('operating_profit_per_share', {})[stock] = op_profit * 10000 / (shares * 10000)  # shares in 万股
            # surplus_reserve_fund_per_share & capital_reserve_fund_per_share: not directly given, could get from balance sheet per share if needed
            # debt ratios, etc.:
            if 'debt_to_equity_ratio' in fund_factors or 'debt_to_equity' in row:
                fund_factors.setdefault('debt_to_equity_ratio', {})[stock] = row.get('debt_to_equity', np.nan)
            if 'debt_to_asset_ratio' in fund_factors or 'debt_to_assets' in row:
                fund_factors.setdefault('debt_to_asset_ratio', {})[stock] = row.get('debt_to_assets', np.nan)
            # invest_income_associates_to_total_profit: not directly; would need income breakdown if available
            # net_operate_cash_flow_to_total_liability: could use ocf_to_liabilities from row
            if 'ocf_to_liabilities' in row:
                fund_factors.setdefault('net_operate_cash_flow_to_total_liability', {})[stock] = row['ocf_to_liabilities']
            # operating_profit_to_total_profit: can compute as operating_profit/total_profit
            if 'opincome_of_ebt' in row:  # operating income/EBT
                fund_factors.setdefault('operating_profit_to_total_profit', {})[stock] = row['opincome_of_ebt']
            # ROA_TTM, ROE_TTM:
            if 'roa' in row:
                fund_factors.setdefault('roa_ttm', {})[stock] = row['roa']
            if 'roe' in row:
                fund_factors.setdefault('roe_ttm', {})[stock] = row['roe']
            # liquidity factor: maybe current_ratio (流动比率)
            if 'current_ratio' in row:
                fund_factors.setdefault('liquidity', {})[stock] = row['current_ratio']
            # beta: will compute separately below
            # book_to_price_ratio: = 1/pb (since pb = price/book)
            if stock in df_factor.index and 'pb' in df_factor.columns:
                pb = df_factor.at[stock, 'pb']
                fund_factors.setdefault('book_to_price_ratio', {})[stock] = (1/pb) if pb else np.nan
            # earnings_to_price_ratio or earnings_yield: use 1/PE (ttm)
            if stock in df_factor.index and 'pe_ttm' in df_factor.columns:
                pe_ttm = df_factor.at[stock, 'pe_ttm']
                fund_factors.setdefault('earnings_yield', {})[stock] = (1/pe_ttm) if pe_ttm else np.nan
            # growth: could use yoy metrics from fina_indicator (e.g., net profit YoY)
            if 'n_income_yoy' in row:
                fund_factors.setdefault('growth', {})[stock] = row['n_income_yoy']
            # momentum: 12-month price momentum
            # computed later after price data (see below)
        else:
            # If no financial data found for this stock (possibly newly listed), fill NaNs
            for key in ['asset_impairment_loss_ttm','cash_flow_to_price_ratio','market_cap',
                        'net_asset_per_share','total_operating_revenue_per_share','net_operate_cash_flow_per_share',
                        'operating_profit_per_share','debt_to_equity_ratio','debt_to_asset_ratio','roe_ttm','roa_ttm',
                        'liquidity','book_to_price_ratio','earnings_yield','growth']:
                fund_factors.setdefault(key, {})[stock] = np.nan
    # Compute beta (60-day) for each stock relative to an index (use CSI300 as benchmark):
    benchmark = '399300.SZ'  # CSI300 index code for Tushare
    # get index prices for last 60 days
    idx_df = pro.index_daily(ts_code=benchmark, end_date=date_str, start_date=(date_dt - timedelta(days=80)).strftime("%Y%m%d"),
                              fields="trade_date, close")
    idx_df.sort_values('trade_date', inplace=True)
    idx_prices = idx_df['close'].astype(float).values
    idx_rets = pd.Series(idx_prices).pct_change().dropna().values
    beta_vals = {}
    for stock in stock_list:
        if stock in ohlcv_dict:
            prices = ohlcv_dict[stock]['close'].astype(float).values
            rets = pd.Series(prices).pct_change().dropna().values
            if len(rets) >= 20 and len(idx_rets) >= len(rets):
                # align lengths
                n = len(rets)
                cov = np.cov(rets[-n:], idx_rets[-n:])[0,1]
                var = np.var(idx_rets[-n:])
                beta_vals[stock] = cov/var if var != 0 else np.nan
            else:
                beta_vals[stock] = np.nan
        else:
            beta_vals[stock] = np.nan
    fund_factors.setdefault('beta', pd.Series(beta_vals))
    # Momentum (12-month price change):
    momentum_vals = {}
    for stock in stock_list:
        if stock in price_data.index:
            prices = price_data.loc[stock].dropna()
            if len(prices) > 250:
                momentum_vals[stock] = prices.iloc[-1] / prices.iloc[0] - 1.0
            else:
                momentum_vals[stock] = np.nan
        else:
            momentum_vals[stock] = np.nan
    fund_factors.setdefault('momentum', pd.Series(momentum_vals))
    # Combine all factors into a DataFrame
    factor_df = pd.DataFrame(index=stock_list)
    # Add technical factors
    for fac, series in factors.items():
        factor_df[fac] = series
    # Add fundamental factors
    for fac, vals in fund_factors.items():
        factor_df[fac] = pd.Series(vals)
    return factor_df
