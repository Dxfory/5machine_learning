import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
import warnings

warnings.filterwarnings("ignore")

from factors import AVAILABLE_FACTORS, get_factor_values, get_price_df, get_stock_pool, get_trade_days


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


class MonthlyRolling5ModelStrategy:
    """5-model Monthly Rolling Training Strategy with probability ensemble."""

    def __init__(self):
        self.random_state = 42
        set_global_seed(self.random_state)

        self.feature_columns = list(AVAILABLE_FACTORS)
        self.jqfactors_list = self.feature_columns.copy()

        self.sample_every_n = 5  # pick every 5th trading day for training samples
        self.forward_days = 20  # forward return window for labels
        self.training_months = 36
        self.min_stock_count = 30
        self.max_missing_ratio = 0.4
        self.winsor_limits = (0.01, 0.99)

        self.models_config: Dict[str, Dict[str, Optional[float]]] = {
            "lgb": {"name": "LightGBM", "model": None, "mse": None, "r2": None, "weight": None},
            "xgb": {"name": "XGBoost", "model": None, "mse": None, "r2": None, "weight": None},
            "svr": {"name": "SVR", "model": None, "scaler": None, "mse": None, "r2": None, "weight": None},
            "rf": {"name": "RandomForest", "model": None, "mse": None, "r2": None, "weight": None},
            "ridge": {"name": "Ridge", "model": None, "mse": None, "r2": None, "weight": None},
        }

        self.selected_features: List[str] = []
        self.last_train_date: Optional[pd.Timestamp] = None
        self.is_trained = False
        self.feature_scaler: Optional[RobustScaler] = None
        self.initial_model_loaded = False
        self.first_trade_date: Optional[pd.Timestamp] = None
        self.training_losses: List[float] = []
        self.validation_history: List[Dict[str, float]] = []
        self.training_model_metrics: List[Dict[str, Dict[str, float]]] = []
        self.episode_count = 0
        self.training_feature_columns: List[str] = self.feature_columns.copy()

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        self.model_filepath = str(models_dir / "5model_monthly_ensemble_mse.pkl")
        self.metadata_filepath = str(models_dir / "5model_metadata_mse.json")

        self.config = {
            "training_months": self.training_months,
            "sample_every_n": self.sample_every_n,
            "forward_days": self.forward_days,
            "min_stock_count": self.min_stock_count,
            "max_missing_ratio": self.max_missing_ratio,
            "winsor_limits": self.winsor_limits,
            "random_state": self.random_state,
        }
        self.minimum_r2 = -0.1

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------
    def monthly_retrain(self, current_date: pd.Timestamp, stock_pool: str = "ZXBZ") -> bool:
        try:
            if self.initial_model_loaded:
                print("Initial model loaded, skipping first retraining cycle")
                self.initial_model_loaded = False
                return False

            if self.last_train_date and current_date.month == self.last_train_date.month and current_date.year == self.last_train_date.year:
                print("Already trained this month, skipping retrain")
                return True

            print("▶ Starting monthly retraining for 5-model ensemble...")
            end_date = (current_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=self.training_months)).strftime("%Y-%m-%d")

            train_data = self.get_training_data(start_date, end_date, stock_pool)
            if train_data is None or len(train_data) < self.min_stock_count:
                print("WARNING: Not enough training data, using existing models")
                return False

            self.training_feature_columns = [col for col in train_data.columns if col != "label"]
            selected_features = self.feature_selection(train_data)
            if len(selected_features) < 5:
                print("WARNING: Too few effective features, falling back to in-sample feature set")
                selected_features = self.training_feature_columns
            selected_features = [col for col in selected_features if col in self.training_feature_columns]
            if not selected_features:
                print("ERROR: No usable features after selection, aborting retrain")
                return False
            self.selected_features = selected_features

            X = train_data[selected_features]
            y = train_data["label"]
            if len(y) == 0 or np.isclose(y.std(ddof=0), 0.0):
                print("WARNING: Training labels lack variation, skip retraining")
                return False

            X_processed = self.robust_data_preprocessing(X)
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed,
                y,
                test_size=0.2,
                random_state=self.random_state,
            )
            print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

            success_count = 0
            model_metrics: Dict[str, Dict[str, float]] = {}

            if self.train_lightgbm(X_train, y_train, X_val, y_val):
                success_count += 1
                model_metrics["lgb"] = {
                    "mse": self.models_config["lgb"]["mse"],
                    "r2": self.models_config["lgb"]["r2"],
                }
            if self.train_xgboost(X_train, y_train, X_val, y_val):
                success_count += 1
                model_metrics["xgb"] = {
                    "mse": self.models_config["xgb"]["mse"],
                    "r2": self.models_config["xgb"]["r2"],
                }
            if self.train_svr(X_train, y_train, X_val, y_val):
                success_count += 1
                model_metrics["svr"] = {
                    "mse": self.models_config["svr"]["mse"],
                    "r2": self.models_config["svr"]["r2"],
                }
            if self.train_random_forest(X_train, y_train, X_val, y_val):
                success_count += 1
                model_metrics["rf"] = {
                    "mse": self.models_config["rf"]["mse"],
                    "r2": self.models_config["rf"]["r2"],
                }
            if self.train_ridge_regression(X_train, y_train, X_val, y_val):
                success_count += 1
                model_metrics["ridge"] = {
                    "mse": self.models_config["ridge"]["mse"],
                    "r2": self.models_config["ridge"]["r2"],
                }

            if success_count >= 3:
                weights_ready = self.calculate_dynamic_weights()
                if not weights_ready:
                    print("WARNING: Unable to derive ensemble weights, retaining previous models.")
                    return False
                self.is_trained = True
                self.last_train_date = current_date
                self.episode_count += 1
                self.training_model_metrics.append(model_metrics)
                validation_metrics = self.evaluate_ensemble(X_val, y_val)
                self.validation_history.append(validation_metrics)
                save_success = self.save_models(selected_features, validation_metrics)
                if save_success:
                    print("✅ Model saved successfully.")
                else:
                    print("❌ Model save failed!")
                print(
                    "✅ Retraining complete! Models trained: "
                    f"{success_count}/5, Ensemble RMSE: {validation_metrics['rmse']:.6f}, R2: {validation_metrics['r2']:.4f}"
                )
                return True

            print("WARNING: Fewer than 3 models trained successfully, keeping existing models.")
            return False
        except Exception as exc:
            print(f"Retraining failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def get_training_data(self, start_date: str, end_date: str, stock_pool: str = "ZXBZ") -> Optional[pd.DataFrame]:
        try:
            trade_days = get_trade_days(start_date, end_date)
            if len(trade_days) <= self.forward_days:
                return None

            sample_indices = list(range(0, len(trade_days) - self.forward_days, self.sample_every_n))
            if not sample_indices:
                return None

            sections: List[pd.DataFrame] = []
            feature_sets: List[set] = []

            for idx in sample_indices:
                trade_day = trade_days[idx]
                forward_idx = idx + self.forward_days
                if forward_idx >= len(trade_days):
                    break
                forward_day = trade_days[forward_idx]

                stocks = get_stock_pool(trade_day, stock_pool)
                if len(stocks) < self.min_stock_count:
                    continue

                factor_df = get_factor_values(stocks, trade_day)
                if factor_df.empty:
                    continue

                factor_df = self.drop_sparse_columns(factor_df)
                if factor_df.empty:
                    continue

                factor_df = self.cross_sectional_standardize(factor_df)

                price_df = get_price_df(factor_df.index.tolist(), trade_day, forward_day)
                if price_df.empty:
                    continue

                start_ts = pd.to_datetime(trade_day)
                end_ts = pd.to_datetime(forward_day)
                forward_returns = {}
                for stock in factor_df.index:
                    if stock not in price_df.index:
                        continue
                    series = price_df.loc[stock].dropna().sort_index()
                    if start_ts not in series.index or end_ts not in series.index:
                        continue
                    start_price = series.loc[start_ts]
                    end_price = series.loc[end_ts]
                    if start_price and end_price and start_price > 0:
                        forward_returns[stock] = end_price / start_price - 1.0

                if len(forward_returns) < self.min_stock_count:
                    continue

                returns_series = pd.Series(forward_returns)
                if len(returns_series) == 0 or np.isclose(returns_series.std(ddof=0), 0.0):
                    continue

                section_df = factor_df.loc[returns_series.index].copy()
                section_df["label"] = returns_series
                sections.append(section_df)
                feature_sets.append(set(section_df.columns) - {"label"})

            if not sections:
                return None

            common_features = set.intersection(*feature_sets) if feature_sets else set()
            ordered_common_features = [col for col in self.feature_columns if col in common_features]
            if not ordered_common_features:
                return None

            aligned_sections = [df[ordered_common_features + ["label"]] for df in sections]
            combined = pd.concat(aligned_sections, axis=0)
            combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return combined
        except Exception as exc:
            print(f"Error constructing training data: {exc}")
            return None

    def drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        missing_ratio = df.isna().mean()
        keep_cols = missing_ratio[missing_ratio <= self.max_missing_ratio].index.tolist()
        return df[keep_cols]

    def cross_sectional_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df_proc = df.replace([np.inf, -np.inf], np.nan)
        lower = df_proc.quantile(self.winsor_limits[0])
        upper = df_proc.quantile(self.winsor_limits[1])
        df_proc = df_proc.clip(lower=lower, upper=upper, axis=1)
        means = df_proc.mean()
        stds = df_proc.std().replace(0, np.nan)
        df_proc = (df_proc - means) / stds
        df_proc = df_proc.fillna(0.0)
        return df_proc

    # ------------------------------------------------------------------
    # Model training helpers
    # ------------------------------------------------------------------
    def robust_data_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        X_proc = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.feature_scaler is None:
            self.feature_scaler = RobustScaler()
            transformed = self.feature_scaler.fit_transform(X_proc)
        else:
            transformed = self.feature_scaler.transform(X_proc)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(transformed, index=X.index, columns=X.columns)

    def _record_model_performance(self, model_name: str, model, y_val: pd.Series, preds: np.ndarray) -> None:
        preds = np.asarray(preds).flatten()
        mse = mean_squared_error(y_val, preds)
        try:
            r2 = r2_score(y_val, preds)
        except ValueError:
            r2 = float("nan")
        self.models_config[model_name]["model"] = model
        self.models_config[model_name]["mse"] = float(mse)
        self.models_config[model_name]["r2"] = float(r2)
        print(f"{self.models_config[model_name]['name']} trained, MSE: {mse:.6f}, R2: {r2:.4f}")

    def train_lightgbm(self, X_train, y_train, X_val, y_val) -> bool:
        try:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbosity": -1,
                "seed": self.random_state,
                "deterministic": True,
                "force_col_wise": True,
            }
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            preds = model.predict(X_val)
            self._record_model_performance("lgb", model, y_val, preds)
            return True
        except Exception as exc:
            print(f"LightGBM training failed: {exc}")
            return False

    def train_xgboost(self, X_train, y_train, X_val, y_val) -> bool:
        try:
            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "rmse",
                "random_state": self.random_state,
                "n_estimators": 300,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            self._record_model_performance("xgb", model, y_val, preds)
            return True
        except Exception as exc:
            print(f"XGBoost training failed: {exc}")
            return False

    def train_svr(self, X_train, y_train, X_val, y_val) -> bool:
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            sample_size = min(5000, len(X_train_scaled))
            if sample_size < len(X_train_scaled):
                idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
                X_train_scaled = X_train_scaled[idx]
                y_train_sampled = y_train.iloc[idx]
            else:
                y_train_sampled = y_train
            model = SVR(kernel="rbf")
            model.fit(X_train_scaled, y_train_sampled)
            preds = model.predict(X_val_scaled)
            self.models_config["svr"]["scaler"] = scaler
            self._record_model_performance("svr", model, y_val, preds)
            return True
        except Exception as exc:
            print(f"SVR training failed: {exc}")
            return False

    def train_random_forest(self, X_train, y_train, X_val, y_val) -> bool:
        try:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            self._record_model_performance("rf", model, y_val, preds)
            return True
        except Exception as exc:
            print(f"RandomForest training failed: {exc}")
            return False

    def train_ridge_regression(self, X_train, y_train, X_val, y_val) -> bool:
        try:
            model = Ridge(alpha=1.0, solver="sag", random_state=self.random_state, max_iter=500)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            self._record_model_performance("ridge", model, y_val, preds)
            return True
        except Exception as exc:
            print(f"Ridge regression training failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Ensemble aggregation
    # ------------------------------------------------------------------
    def calculate_dynamic_weights(self) -> bool:
        valid_entries = []
        fallback_entries = []
        for name, cfg in self.models_config.items():
            model = cfg.get("model")
            mse = cfg.get("mse")
            if model is None or mse is None:
                continue
            r2 = cfg.get("r2")
            if r2 is not None and not np.isnan(r2) and r2 < self.minimum_r2:
                print(f"{cfg['name']} filtered out due to low R2 ({r2:.4f})")
                fallback_entries.append((name, mse))
                continue
            valid_entries.append((name, mse))

        if not valid_entries and fallback_entries:
            fallback_entries.sort(key=lambda item: item[1])
            valid_entries = fallback_entries
            print("All models below R2 threshold; using best models by MSE for weighting.")

        if not valid_entries:
            print("❌ No valid models to weight.")
            for cfg in self.models_config.values():
                cfg["weight"] = None
            return False

        mse_values = np.array([mse for _, mse in valid_entries])
        mse_clip_low = np.percentile(mse_values, 5)
        mse_clip_high = np.percentile(mse_values, 95)
        mse_values = np.clip(mse_values, mse_clip_low, mse_clip_high)
        temperature = 5.0
        weights_raw = np.exp(-temperature * mse_values)
        weights = weights_raw / weights_raw.sum()

        for (name, _), weight in zip(valid_entries, weights):
            self.models_config[name]["weight"] = float(weight)
            print(f"{self.models_config[name]['name']} weight: {weight:.4f} (MSE: {self.models_config[name]['mse']:.6f})")
        for name, cfg in self.models_config.items():
            if name not in {entry[0] for entry in valid_entries}:
                cfg["weight"] = None
        return True

    def evaluate_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        preds = []
        weights = []
        used_models = []
        for name, cfg in self.models_config.items():
            if cfg.get("model") is None or cfg.get("weight") is None:
                continue
            pred = self._predict_single(cfg, X_val)
            if pred is None:
                continue
            preds.append(pred)
            weights.append(cfg["weight"])
            used_models.append(cfg["name"])
        if not preds:
            return {"rmse": float("nan"), "r2": float("nan")}
        ensemble_pred = np.average(np.vstack(preds), axis=0, weights=weights)
        mse = mean_squared_error(y_val, ensemble_pred)
        rmse = float(np.sqrt(mse))
        try:
            r2 = float(r2_score(y_val, ensemble_pred))
        except ValueError:
            r2 = float("nan")
        print(
            "Ensemble evaluation – models used: "
            f"{used_models}, weights: {[round(w,4) for w in weights]}, RMSE: {rmse:.6f}, R2: {r2:.4f}"
        )
        return {"rmse": rmse, "r2": r2}

    def _predict_single(self, cfg: Dict[str, Any], X: pd.DataFrame) -> Optional[np.ndarray]:
        model = cfg.get("model")
        if model is None:
            return None
        name = cfg.get("name")
        if name == "LightGBM":
            return model.predict(X)
        if name == "XGBoost":
            return model.predict(X)
        if name == "SVR":
            scaler = cfg.get("scaler")
            if scaler is None:
                return None
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)
        if name in {"RandomForest", "Ridge"}:
            return model.predict(X)
        return None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or not self.selected_features:
            print("Model not trained, returning random predictions")
            return np.random.random(len(X))
        X_aligned = self.prepare_prediction_features(X)
        if X_aligned.empty:
            return np.zeros(len(X))
        X_proc = self.robust_data_preprocessing(X_aligned)
        preds = []
        weights = []
        for cfg in self.models_config.values():
            if cfg.get("model") is None or cfg.get("weight") is None:
                continue
            pred = self._predict_single(cfg, X_proc)
            if pred is None:
                continue
            preds.append(pred)
            weights.append(cfg["weight"])
        if not preds:
            return np.zeros(len(X_proc))
        return np.average(np.vstack(preds), axis=0, weights=weights)

    def prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not self.selected_features:
            return pd.DataFrame(index=df.index if not df.empty else [])
        df_proc = df.copy()
        for col in self.selected_features:
            if col not in df_proc.columns:
                df_proc[col] = np.nan
        df_proc = df_proc[self.selected_features]
        df_proc = self.cross_sectional_standardize(df_proc)
        df_proc = df_proc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df_proc

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_models(
        self, selected_features: Sequence[str], validation_metrics: Dict[str, float], filepath: Optional[str] = None
    ) -> bool:
        if filepath is None:
            filepath = self.model_filepath
        try:
            model_data = {
                "models_config": self.models_config,
                "selected_features": list(selected_features),
                "feature_scaler": self.feature_scaler,
                "validation_metrics": validation_metrics,
                "saved_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_losses": self.training_losses,
                "validation_history": self.validation_history,
                "training_model_metrics": self.training_model_metrics,
                "episode_count": self.episode_count,
                "last_train_date": self.last_train_date.strftime("%Y-%m-%d") if self.last_train_date else None,
                "config": self.config,
            }
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            metadata = {
                "training_losses": self.training_losses,
                "validation_history": self.validation_history,
                "training_model_metrics": self.training_model_metrics,
                "episode_count": self.episode_count,
                "last_train_date": self.last_train_date.strftime("%Y-%m-%d") if self.last_train_date else None,
                "save_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selected_features_count": len(selected_features),
                "current_validation_metrics": validation_metrics,
                "config": self.config,
            }
            with open(self.metadata_filepath, "w") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            print(f"Model save failed: {exc}")
            return False

    def load_models(self, filepath: Optional[str] = None) -> bool:
        if filepath is None:
            filepath = self.model_filepath
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
            self.models_config = model_data["models_config"]
            self.selected_features = model_data["selected_features"]
            self.feature_scaler = model_data.get("feature_scaler")
            self.is_trained = True
            self.training_losses = model_data.get("training_losses", [])
            self.validation_history = model_data.get(
                "validation_history", model_data.get("training_auc_scores", [])
            )
            self.training_model_metrics = model_data.get(
                "training_model_metrics", model_data.get("training_model_mses", [])
            )
            self.episode_count = model_data.get("episode_count", 0)
            last_date_str = model_data.get("last_train_date")
            if last_date_str:
                self.last_train_date = pd.to_datetime(last_date_str)
            self.config = model_data.get("config", self.config)
            self.initial_model_loaded = True
            print(f"✅ Loaded saved model ensemble from {filepath}")
            print(f"Features: {len(self.selected_features)}, past trainings: {self.episode_count}")
            return True
        except Exception as exc:
            print(f"Model load failed: {exc}")
            return False

    def manual_save_models(self) -> bool:
        if not self.is_trained:
            print("No trained model to save.")
            return False
        try:
            print("=" * 60)
            print("Manual save of 5-model ensemble")
            print("=" * 60)
            current_metrics = self.validation_history[-1] if self.validation_history else {"rmse": float("nan"), "r2": float("nan")}
            success = self.save_models(self.selected_features, current_metrics)
            if success:
                print("✅ Manual save successful.")
            else:
                print("❌ Manual save failed.")
            return success
        except Exception as exc:
            print(f"Manual save error: {exc}")
            return False

    def feature_selection(self, df: pd.DataFrame) -> List[str]:
        if "label" not in df.columns:
            return self.feature_columns
        features_only = df.drop(columns=["label"])
        if features_only.empty:
            return self.feature_columns
        corr_matrix = features_only.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
        selected = [col for col in features_only.columns if col not in to_drop]
        return selected if selected else self.feature_columns
