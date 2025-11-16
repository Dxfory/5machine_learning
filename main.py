from jqdata import *
from jqfactor import get_factor_values
import datetime
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import json
warnings.filterwarnings('ignore')


class MonthlyRolling5ModelStrategy:
    """5-model monthly rolling training strategy - weights allocated by MSE"""

    def __init__(self):
        # Factor list
        self.jqfactors_list = [
            'asset_impairment_loss_ttm', 'cash_flow_to_price_ratio', 'market_cap',
            'interest_free_current_liability', 'EBITDA', 'financial_assets',
            'gross_profit_ttm', 'net_working_capital', 'non_recurring_gain_loss', 'EBIT',
            'sales_to_price_ratio', 'AR', 'ARBR', 'ATR6', 'DAVOL10', 'MAWVAD', 'TVMA6',
            'PSY', 'VOL10', 'VDIFF', 'VEMA26', 'VMACD', 'VOL120', 'VOSC', 'VR', 'WVAD',
            'arron_down_25', 'arron_up_25', 'BBIC', 'MASS', 'Rank1M', 'single_day_VPT',
            'single_day_VPT_12', 'single_day_VPT_6', 'Volume1M',
            'capital_reserve_fund_per_share', 'net_asset_per_share',
            'net_operate_cash_flow_per_share', 'operating_profit_per_share',
            'total_operating_revenue_per_share', 'surplus_reserve_fund_per_share',
            'ACCA', 'account_receivable_turnover_days', 'account_receivable_turnover_rate',
            'adjusted_profit_to_total_profit', 'super_quick_ratio', 'MLEV',
            'debt_to_equity_ratio', 'debt_to_tangible_equity_ratio',
            'equity_to_fixed_asset_ratio', 'fixed_asset_ratio', 'intangible_asset_ratio',
            'invest_income_associates_to_total_profit', 'long_debt_to_asset_ratio',
            'long_debt_to_working_capital_ratio', 'net_operate_cash_flow_to_total_liability',
            'net_operating_cash_flow_coverage', 'non_current_asset_ratio',
            'operating_profit_to_total_profit', 'roa_ttm', 'roe_ttm', 'Kurtosis120',
            'Kurtosis20', 'Kurtosis60', 'sharpe_ratio_20', 'sharpe_ratio_60',
            'Skewness120', 'Skewness20', 'Skewness60', 'Variance120', 'Variance20',
            'liquidity', 'beta', 'book_to_price_ratio', 'cash_earnings_to_price_ratio',
            'cube_of_size', 'earnings_to_price_ratio', 'earnings_yield', 'growth',
            'momentum', 'natural_log_of_market_cap', 'boll_down', 'MFI14', 'MAC10',
            'fifty_two_week_close_rank', 'price_no_fq'
        ]

        # Configuration of 5 models
        self.models_config = {
            'lgb': {'name': 'LightGBM', 'model': None, 'mse': None, 'weight': None},
            'xgb': {'name': 'XGBoost', 'model': None, 'mse': None, 'weight': None},
            'svr': {'name': 'SVR', 'model': None, 'scaler': None, 'mse': None, 'weight': None},
            'rf': {'name': 'RandomForest', 'model': None, 'mse': None, 'weight': None},
            'lr': {'name': 'LinearRegression', 'model': None, 'mse': None, 'weight': None}
        }

        # Rolling MSE window (store at most last 3 rounds)
        self.rolling_mses = {model_name: [] for model_name in self.models_config.keys()}

        self.selected_features = []
        self.last_train_date = None
        self.is_trained = False
        self.feature_scaler = None
        self.training_months = 36
        self.initial_model_loaded = False
        self.first_trade_date = None

        # File paths
        self.model_filepath = '5model_monthly_ensemble_mse.pkl'
        self.metadata_filepath = '5model_metadata_mse.json'

        # Training records
        self.training_losses = []
        self.training_auc_scores = []
        self.training_model_mses = []
        self.episode_count = 0
        
    def get_dynamic_training_months(self, end_date):
        """
        Dynamically adjust training window length based on HS300 volatility
        over the last 30 trading days:
        - Daily volatility > 2%: 24 months (recent regime more important)
        - Daily volatility < 1%: 48 months (need longer history)
        - Otherwise: default self.training_months (36 months)
        """
        try:
            index_data = get_price(
                '000300.XSHG',
                end_date=end_date,
                count=30,
                frequency='daily',
                fields=['close']
            )
            if index_data is None or len(index_data) < 2:
                log.warn("Failed to get HS300 data, use default training window")
                return self.training_months

            close = index_data['close']
            ret = close.pct_change().dropna()
            if len(ret) == 0:
                log.warn("HS300 return series is empty, use default training window")
                return self.training_months

            vol = ret.std()
            log.info(f"HS300 30-day daily volatility: {vol:.4%}")

            if vol > 0.02:
                months = 24
            elif vol < 0.01:
                months = 48
            else:
                months = self.training_months

            log.info(f"Dynamic training window months: {months}")
            return months

        except Exception as e:
            log.warn(f"Failed to compute dynamic training window: {e}, use default value")
            return self.training_months

    def monthly_retrain(self, context):
        """Monthly retraining function - supports initial training"""
        try:
            # If model is loaded initially and no initial training is required, skip
            if self.initial_model_loaded and not hasattr(g, 'need_initial_training'):
                log.info("Model loaded from file, skip initial training")
                self.initial_model_loaded = False
                return False

            current_date = context.current_dt

            # For the first training, skip month check
            if hasattr(g, 'need_initial_training') and g.need_initial_training:
                log.info("Perform initial model training, skip month check")
            else:
                if self.last_train_date and current_date.month == self.last_train_date.month:
                    log.info("Already trained this month, skip retraining")
                    return True

            log.info("Start 5-model monthly retraining (MSE-weighted)...")

            # Get training data with dynamic training window
            end_date = context.previous_date
            dynamic_months = self.get_dynamic_training_months(end_date)
            start_date = (end_date - datetime.timedelta(days=dynamic_months * 30)).strftime('%Y-%m-%d')
            train_data = self.get_training_data(start_date, end_date.strftime('%Y-%m-%d'))

            if train_data is None or len(train_data) < 100:
                log.warn("Training data is insufficient, keep current model")
                return False

            # Feature selection
            selected_features = self.feature_selection(train_data)
            if len(selected_features) < 10:
                log.warn("Too few valid features, fallback to existing feature list")
                selected_features = self.selected_features if self.selected_features else self.jqfactors_list[:30]

            log.info(f"Training with {len(selected_features)} features")

            # Build features and targets: binary / ternary / regression targets
            X = train_data[selected_features]

            # Original binary label (backward compatible with previous training)
            y_binary = train_data['label']

            # Regression target: future returns
            if 'ret' in train_data.columns:
                y_reg = train_data['ret']
            else:
                # Extreme fallback: if ret is missing, use label as pseudo regression target
                y_reg = y_binary.astype(float)

            # Ternary label: if not provided, recompute with global quantiles
            if 'label3' in train_data.columns:
                y_cls3 = train_data['label3']
            else:
                q25_global = y_reg.quantile(0.25)
                q75_global = y_reg.quantile(0.75)

                def to_label3_global(x):
                    if x < q25_global:
                        return 0
                    elif x <= q75_global:
                        return 1
                    else:
                        return 2

                y_cls3 = y_reg.apply(to_label3_global)

            X_processed = self.robust_data_preprocessing(X)

            # Train/validation split for: binary label, regression target, and ternary label
            X_train, X_val, y_bin_train, y_bin_val, y_reg_train, y_reg_val, y_cls3_train, y_cls3_val = train_test_split(
                X_processed, y_binary, y_reg, y_cls3,
                test_size=0.2, random_state=42, stratify=y_cls3
            )
            log.info(f"Training data shape: {X_train.shape}, validation data shape: {X_val.shape}")

            # Train 5 models
            success_count = 0
            model_mses = {}
            if self.train_lightgbm(X_train, y_bin_train, X_val, y_bin_val, model_mses):
                success_count += 1
            if self.train_xgboost(X_train, y_bin_train, X_val, y_bin_val, model_mses):
                success_count += 1
            if self.train_svr(X_train, y_bin_train, X_val, y_bin_val, model_mses):
                success_count += 1
            if self.train_random_forest(X_train, y_bin_train, X_val, y_bin_val, model_mses):
                success_count += 1
            if self.train_linear_regression(X_train, y_bin_train, X_val, y_bin_val, model_mses):
                success_count += 1

            if success_count >= 3:
                # Re-compute combined loss using regression + ternary classification
                model_losses = self.compute_model_losses(X_val, y_reg_val, y_cls3_val)
                if not model_losses:
                    log.warn("Failed to compute combined loss, fallback to original MSE")
                    model_losses = model_mses  # Fallback to original binary MSE

                # Save combined loss into config and update rolling window
                for model_name, loss in model_losses.items():
                    if model_name not in self.models_config:
                        continue
                    # mse field now stores combined loss
                    self.models_config[model_name]['mse'] = loss

                    if model_name not in self.rolling_mses:
                        self.rolling_mses[model_name] = []
                    self.rolling_mses[model_name].append(loss)
                    if len(self.rolling_mses[model_name]) > 3:
                        self.rolling_mses[model_name].pop(0)

                # Compute dynamic weights based on rolling combined loss + penalty
                self.calculate_dynamic_weights()

                self.selected_features = selected_features
                self.is_trained = True
                self.last_train_date = current_date
                self.episode_count += 1
                self.training_model_mses.append(model_losses)

                # Evaluate ensemble; here we use ternary accuracy as overall score
                auc_score = self.evaluate_ensemble(X_val, y_cls3_val, y_reg_val)
                self.training_auc_scores.append(auc_score)

                # Save models
                save_success = self.save_models(selected_features, auc_score)
                if save_success:
                    log.info("✅ Models saved successfully")
                else:
                    log.error("❌ Failed to save models")

                log.info(
                    f"✅ 5-model monthly retraining finished! "
                    f"Successful models: {success_count}/5, AUC (3-class acc): {auc_score:.4f}"
                )
                return True
            else:
                log.warn("Not enough successfully trained models, keep existing models")
                return False

        except Exception as e:
            log.warn(f"Monthly retraining failed: {e}")
            return False

    def calculate_dynamic_weights(self):
        """
        Compute dynamic weights based on each model's
        [rolling MSE mean + penalty term].

        Weight formula:
        weight_i = [1/(rolling_mse_i + eps) * penalty_i] / Σ_k [1/(rolling_mse_k + eps) * penalty_k]

        penalty_i:
            - Normal model: 1.0
            - Low-contribution model: 0.5
              (if the last 2 rolling MSE values are both significantly above the global average)
        """
        valid_models = []
        for model_name, config in self.models_config.items():
            if config['model'] is not None and config['mse'] is not None:
                history = self.rolling_mses.get(model_name, [])
                # Rolling MSE: use average of history if exists, otherwise use current loss
                if history:
                    rolling_mse = float(np.mean(history))
                else:
                    rolling_mse = float(config['mse'])
                valid_models.append((model_name, rolling_mse, history))

        if not valid_models:
            log.error("No valid model, cannot compute weights")
            return

        # Print rolling MSE history
        log.info("Rolling MSE history for this round:")
        for model_name, history in self.rolling_mses.items():
            if history:
                hist_str = ["%.4f" % x for x in history]
                log.info(f"{model_name}: hist={hist_str}")

        # Global mean of rolling MSE, used as penalty threshold baseline
        rolling_mse_list = [rmse for _, rmse, _ in valid_models]
        global_mean_mse = float(np.mean(rolling_mse_list))
        threshold = 1.5 * global_mean_mse  # penalty threshold

        epsilon = 1e-8
        numerators = []

        for model_name, rolling_mse, history in valid_models:
            # Default: no penalty
            penalty = 1.0
            # If model has at least 2 records and the last 2 MSEs are both > threshold,
            # it is treated as a low-contribution model
            if len(history) >= 2 and history[-1] > threshold and history[-2] > threshold:
                penalty = 0.5

            numer = 1.0 / (rolling_mse + epsilon) * penalty
            numerators.append((model_name, numer, rolling_mse, penalty))

        total_numer = sum(n for _, n, _, _ in numerators)
        if total_numer <= 0:
            # Fallback in extreme case: equal weight
            equal_w = 1.0 / len(numerators)
            for model_name, _, rolling_mse, penalty in numerators:
                self.models_config[model_name]['weight'] = equal_w
                log.info(
                    f"{self.models_config[model_name]['name']} weight (equal-weight fallback): {equal_w:.4f} "
                    f"(rollingMSE: {rolling_mse:.6f}, penalty: {penalty})"
                )

            # Print weights for this round
            log.info("Model weights for this round:")
            for model_name, cfg in self.models_config.items():
                if cfg.get('weight') is not None:
                    log.info(f"{model_name}: weight={cfg['weight']:.4f}, loss={cfg['mse']:.6f}")
            return

        # Normal case: normalize numerators to weights
        for model_name, numer, rolling_mse, penalty in numerators:
            w = numer / total_numer
            self.models_config[model_name]['weight'] = w
            log.info(
                f"{self.models_config[model_name]['name']} weight: {w:.4f} "
                f"(rollingMSE: {rolling_mse:.6f}, penalty: {penalty})"
            )

        # Print weights for this round
        log.info("Model weights for this round:")
        for model_name, cfg in self.models_config.items():
            if cfg.get('weight') is not None:
                log.info(f"{model_name}: weight={cfg['weight']:.4f}, loss={cfg['mse']:.6f}")

    def compute_model_losses(self, X_val, y_reg_val, y_cls3_val):
        """
        Compute combined loss for each model based on:
        loss = 0.4 * MSE_reg + 0.6 * (1 - acc_cls)

        Regression target: future return (ret, normalized to [0,1])
        3-class target: label3 (0 / 1 / 2)
        """
        model_losses = {}
        try:
            # Convert targets to numpy arrays
            if isinstance(y_reg_val, pd.Series):
                y_reg = y_reg_val.values
            else:
                y_reg = np.asarray(y_reg_val)

            if isinstance(y_cls3_val, pd.Series):
                y_cls3 = y_cls3_val.values
            else:
                y_cls3 = np.asarray(y_cls3_val)

            # Normalize returns to [0, 1] to stabilize MSE scale
            ret_min, ret_max = y_reg.min(), y_reg.max()
            if ret_max - ret_min < 1e-8:
                y_reg_norm = np.zeros_like(y_reg)
            else:
                y_reg_norm = (y_reg - ret_min) / (ret_max - ret_min)

            # Use quantiles of normalized returns to define 25% / 75% thresholds
            # for 3-class labels
            q25 = np.percentile(y_reg_norm, 25)
            q75 = np.percentile(y_reg_norm, 75)

            def to_label3_norm(v):
                if v < q25:
                    return 0
                elif v <= q75:
                    return 1
                else:
                    return 2

            for model_name, cfg in self.models_config.items():
                model = cfg.get('model')
                if model is None:
                    continue

                # Prediction behavior consistent with evaluate_ensemble / predict
                if model_name == 'lgb':
                    pred = model.predict(X_val)
                    pred_norm = pred
                elif model_name == 'xgb' and hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]
                    pred_norm = pred
                elif model_name == 'svr':
                    scaler = cfg.get('scaler')
                    if scaler is None:
                        continue
                    X_val_scaled = scaler.transform(X_val)
                    pred = model.predict(X_val_scaled)
                    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                elif model_name in ['rf', 'lr']:
                    pred = model.predict(X_val)
                    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                else:
                    continue

                # Regression MSE on normalized returns
                mse_reg = mean_squared_error(y_reg_norm, pred_norm)

                # 3-class accuracy
                y_pred_cls = np.array([to_label3_norm(v) for v in pred_norm])
                acc_cls = (y_pred_cls == y_cls3).mean()

                combined_loss = 0.4 * mse_reg + 0.6 * (1.0 - acc_cls)
                model_losses[model_name] = combined_loss

                log.info(
                    f"{cfg['name']} - Regression MSE: {mse_reg:.6f}, "
                    f"3-class Acc: {acc_cls:.4f}, Combined loss: {combined_loss:.6f}"
                )

            return model_losses

        except Exception as e:
            log.warn(f"Failed to compute combined model losses: {e}")
            return {}

    def robust_data_preprocessing(self, X):
        """Robust data preprocessing + support for changing feature dimensions"""
        try:
            X_processed = X.copy()
            X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

            # Fill missing values with median
            for col in X_processed.columns:
                if X_processed[col].isnull().any():
                    median_val = X_processed[col].median()
                    X_processed[col].fillna(median_val, inplace=True)

            # Check whether we need to refit the scaler
            need_refit = False
            if self.feature_scaler is None:
                need_refit = True
            else:
                try:
                    n_in = getattr(self.feature_scaler, 'n_features_in_', None)
                    # 1) Old model may not have n_features_in_
                    # 2) Or feature count changed, which means feature structure changed
                    if n_in is None or n_in != X_processed.shape[1]:
                        need_refit = True
                except Exception:
                    need_refit = True

            if need_refit:
                log.info(f"Rebuild/refit feature scaler: n_features={X_processed.shape[1]}")
                self.feature_scaler = RobustScaler()
                X_scaled = self.feature_scaler.fit_transform(X_processed)
            else:
                X_scaled = self.feature_scaler.transform(X_processed)

            # Extra safety checks
            X_scaled = np.where(np.isnan(X_scaled), 0, X_scaled)
            X_scaled = np.where(np.isinf(X_scaled), 0, X_scaled)
            X_scaled = np.clip(X_scaled, -10, 10)

            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        except Exception as e:
            log.warn(f"Data preprocessing failed: {e}")
            return X.fillna(0).replace([np.inf, -np.inf], 0)

    def train_lightgbm(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbosity': -1,
                'random_state': 42
            }

            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            model_mses['lgb'] = mse
            log.info(f"LightGBM training finished, MSE: {mse:.6f}")

            self.models_config['lgb']['model'] = model
            return True

        except Exception as e:
            log.warn(f"LightGBM training failed: {e}")
            return False

    def train_xgboost(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            params = {
                'objective': 'binary:logistic',
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_estimators': 300
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict_proba(X_val)[:, 1]
            mse = mean_squared_error(y_val, y_pred)
            model_mses['xgb'] = mse
            log.info(f"XGBoost training finished, MSE: {mse:.6f}")

            self.models_config['xgb']['model'] = model
            return True

        except Exception as e:
            log.warn(f"XGBoost training failed: {e}")
            return False

    def train_svr(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            sample_size = min(5000, len(X_train_scaled))
            if sample_size < len(X_train_scaled):
                indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
                X_train_sampled = X_train_scaled[indices]
                y_train_sampled = y_train.iloc[indices]
            else:
                X_train_sampled = X_train_scaled
                y_train_sampled = y_train

            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(X_train_sampled, y_train_sampled)

            y_pred = model.predict(X_val_scaled)
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
            mse = mean_squared_error(y_val, y_pred_norm)
            model_mses['svr'] = mse
            log.info(f"SVR training finished, MSE: {mse:.6f}")

            self.models_config['svr']['model'] = model
            self.models_config['svr']['scaler'] = scaler
            return True

        except Exception as e:
            log.warn(f"SVR training failed: {e}")
            return False

    def train_random_forest(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
            mse = mean_squared_error(y_val, y_pred_norm)
            model_mses['rf'] = mse
            log.info(f"RandomForest training finished, MSE: {mse:.6f}")

            self.models_config['rf']['model'] = model
            return True

        except Exception as e:
            log.warn(f"RandomForest training failed: {e}")
            return False

    def train_linear_regression(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
            mse = mean_squared_error(y_val, y_pred_norm)
            model_mses['lr'] = mse
            log.info(f"LinearRegression training finished, MSE: {mse:.6f}")

            self.models_config['lr']['model'] = model
            return True

        except Exception as e:
            log.warn(f"LinearRegression training failed: {e}")
            return False

    def evaluate_ensemble(self, X_val, y_cls3_val, y_reg_val):
        """Evaluate ensemble model: output regression MSE + 3-class accuracy; return accuracy"""
        try:
            predictions = []
            weights = []
            model_names = []

            for model_name, config in self.models_config.items():
                if config['model'] is not None and config.get('weight') is not None:
                    if model_name == 'lgb':
                        pred = config['model'].predict(X_val)
                        pred_norm = pred
                    elif model_name == 'xgb' and hasattr(config['model'], 'predict_proba'):
                        pred = config['model'].predict_proba(X_val)[:, 1]
                        pred_norm = pred
                    elif model_name == 'svr':
                        X_val_scaled = config['scaler'].transform(X_val)
                        pred = config['model'].predict(X_val_scaled)
                        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    elif model_name == 'rf':
                        pred = config['model'].predict(X_val)
                        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    elif model_name == 'lr':
                        pred = config['model'].predict(X_val)
                        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    else:
                        continue

                    predictions.append(pred_norm)
                    weights.append(config['weight'])
                    model_names.append(config['name'])

            if len(predictions) == 0:
                return 0.5

            ensemble_pred = np.average(predictions, axis=0, weights=weights)

            # True returns / labels
            if isinstance(y_reg_val, pd.Series):
                y_reg = y_reg_val.values
            else:
                y_reg = np.asarray(y_reg_val)

            if isinstance(y_cls3_val, pd.Series):
                y_cls3 = y_cls3_val.values
            else:
                y_cls3 = np.asarray(y_cls3_val)

            # Normalize returns
            ret_min, ret_max = y_reg.min(), y_reg.max()
            if ret_max - ret_min < 1e-8:
                y_reg_norm = np.zeros_like(y_reg)
            else:
                y_reg_norm = (y_reg - ret_min) / (ret_max - ret_min)

            # Regression MSE
            mse_reg = mean_squared_error(y_reg_norm, ensemble_pred)

            # 3-class accuracy based on quantiles of normalized returns
            q25 = np.percentile(y_reg_norm, 25)
            q75 = np.percentile(y_reg_norm, 75)

            def to_label3_norm(v):
                if v < q25:
                    return 0
                elif v <= q75:
                    return 1
                else:
                    return 2

            y_pred_cls = np.array([to_label3_norm(v) for v in ensemble_pred])
            acc_cls = (y_pred_cls == y_cls3).mean()

            log.info(
                f"5-model ensemble evaluation - models used: {model_names}, "
                f"Regression MSE: {mse_reg:.6f}, 3-class Acc: {acc_cls:.4f}"
            )
            return acc_cls

        except Exception as e:
            log.warn(f"Ensemble evaluation failed: {e}")
            return 0.5

    def predict(self, X):
        """Predict using dynamic-weight ensemble"""
        try:
            if not self.is_trained or not self.selected_features:
                log.warn("Model not trained, return random predictions")
                return np.random.random(len(X))

            X_processed = self.robust_data_preprocessing(X)

            predictions = []
            weights = []

            for model_name, config in self.models_config.items():
                if config['model'] is not None and config.get('weight') is not None:
                    if model_name == 'lgb':
                        pred = config['model'].predict(X_processed)
                    elif model_name == 'xgb' and hasattr(config['model'], 'predict_proba'):
                        pred = config['model'].predict_proba(X_processed)[:, 1]
                    elif model_name == 'svr':
                        X_processed_scaled = config['scaler'].transform(X_processed)
                        pred = config['model'].predict(X_processed_scaled)
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    elif model_name == 'rf':
                        pred = config['model'].predict(X_processed)
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    elif model_name == 'lr':
                        pred = config['model'].predict(X_processed)
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    else:
                        continue

                    predictions.append(pred)
                    weights.append(config['weight'])

            if len(predictions) == 0:
                return np.zeros(len(X))

            return np.average(predictions, axis=0, weights=weights)

        except Exception as e:
            log.warn(f"Prediction failed: {e}")
            return np.random.random(len(X))

    def save_models(self, selected_features, auc_score, filepath=None):
        """Save models to file"""
        try:
            if filepath is None:
                filepath = self.model_filepath

            model_data = {
                'models_config': self.models_config,
                'selected_features': selected_features,
                'feature_scaler': self.feature_scaler,
                'auc_score': auc_score,
                'saved_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'training_losses': self.training_losses,
                'training_auc_scores': self.training_auc_scores,
                'training_model_mses': self.training_model_mses,
                'episode_count': self.episode_count,
                'last_train_date': self.last_train_date.strftime('%Y-%m-%d') if self.last_train_date else None,
                'rolling_mses': self.rolling_mses
            }

            content = pickle.dumps(model_data)
            write_file(filepath, content)
            log.info(f"Model saved to: {filepath}")

            metadata = {
                'training_losses': self.training_losses,
                'training_auc_scores': self.training_auc_scores,
                'training_model_mses': self.training_model_mses,
                'episode_count': self.episode_count,
                'last_train_date': self.last_train_date.strftime('%Y-%m-%d') if self.last_train_date else None,
                'save_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'selected_features_count': len(selected_features),
                'current_auc': auc_score
            }

            metadata_content = json.dumps(metadata, ensure_ascii=False, indent=2)
            write_file(self.metadata_filepath, metadata_content)
            log.info(f"Metadata saved to: {self.metadata_filepath}")

            return True

        except Exception as e:
            log.error(f"Failed to save model: {str(e)}")
            return False

    def load_models(self, filepath=None):
        """Load models from file"""
        try:
            if filepath is None:
                filepath = self.model_filepath

            content = read_file(filepath)
            model_data = pickle.loads(content)
            log.info(f"Model loaded from: {filepath}")

            self.models_config = model_data['models_config']
            self.selected_features = model_data['selected_features']
            self.feature_scaler = model_data.get('feature_scaler')
            self.is_trained = True

            self.training_losses = model_data.get('training_losses', [])
            self.training_auc_scores = model_data.get('training_auc_scores', [])
            self.training_model_mses = model_data.get('training_model_mses', [])
            self.episode_count = model_data.get('episode_count', 0)

            last_train_date_str = model_data.get('last_train_date')
            if last_train_date_str:
                self.last_train_date = datetime.datetime.strptime(last_train_date_str, '%Y-%m-%d')

            # Restore rolling MSE window (compatible with old model files)
            loaded_rolling = model_data.get('rolling_mses')
            if loaded_rolling is None:
                self.rolling_mses = {model_name: [] for model_name in self.models_config.keys()}
            else:
                self.rolling_mses = {
                    model_name: loaded_rolling.get(model_name, [])
                    for model_name in self.models_config.keys()
                }

            self.is_trained = True
            self.initial_model_loaded = True
            log.info(f"✅ 5-model ensemble system loaded successfully: {filepath}")
            log.info(f"Number of features: {len(self.selected_features)}, "
                     f"training episodes: {self.episode_count}")
            return True

        except Exception as e:
            log.error(f"Failed to load model: {str(e)}")
            return False

    def manual_save_models(self):
        """Manually save models"""
        if not self.is_trained:
            log.warn("Model not trained, cannot save")
            return False

        try:
            log.info("=" * 60)
            log.info("Manual save of 5-model ensemble system")
            log.info("=" * 60)

            current_auc = self.training_auc_scores[-1] if self.training_auc_scores else 0.5
            save_success = self.save_models(self.selected_features, current_auc)

            if save_success:
                log.info("✅ Manual save of 5-model ensemble system completed")
                return True
            else:
                log.error("Manual save failed")
                return False

        except Exception as e:
            log.error(f"Exception during manual save: {str(e)}")
            return False

    def get_training_data(self, start_date, end_date, stock_pool='ZXBZ'):
        try:
            all_dates = get_trade_days(start_date=start_date, end_date=end_date)
            monthly_dates = [pd.to_datetime(date) for date in all_dates
                             if pd.to_datetime(date).day in [1, 15]]

            if len(monthly_dates) < 2:
                return None

            all_data = []

            for i in range(len(monthly_dates) - 1):
                date = monthly_dates[i]
                next_date = monthly_dates[i + 1]

                try:
                    stocks = self.get_stock_pool(date.strftime('%Y-%m-%d'), stock_pool)
                    if len(stocks) == 0:
                        continue

                    factor_data = get_factor_values(
                        stocks, self.jqfactors_list,
                        end_date=date.strftime('%Y-%m-%d'), count=1
                    )
                    df = pd.DataFrame(index=stocks)

                    for factor in self.jqfactors_list:
                        if factor in factor_data:
                            df[factor] = factor_data[factor].iloc[0, :]

                    if df.empty:
                        continue

                    df = df.dropna()
                    if df.empty:
                        continue

                    valid_stocks = []
                    returns = []

                    for stock in df.index:
                        try:
                            price_data = get_price(
                                stock, date.strftime('%Y-%m-%d'),
                                next_date.strftime('%Y-%m-%d'), 'daily',
                                fields=['close'], skip_paused=True
                            )
                            if len(price_data) >= 2:
                                start_price = price_data['close'].iloc[0]
                                end_price = price_data['close'].iloc[-1]
                                stock_return = end_price / start_price - 1
                                valid_stocks.append(stock)
                                returns.append(stock_return)
                        except:
                            continue

                    if len(valid_stocks) < 10:
                        continue

                    df = df.loc[valid_stocks]

                    # Regression target: future returns (continuous)
                    df['ret'] = returns

                    # Binary label (keep original logic for backward compatibility)
                    median_ret = df['ret'].median()
                    df['label'] = np.where(df['ret'] >= median_ret, 1, 0)

                    # 3-class label: based on 25% / 75% quantile
                    q25 = df['ret'].quantile(0.25)
                    q75 = df['ret'].quantile(0.75)

                    def to_label3(x):
                        if x < q25:
                            return 0
                        elif x <= q75:
                            return 1
                        else:
                            return 2

                    df['label3'] = df['ret'].apply(to_label3)

                    all_data.append(df)

                except Exception as e:
                    log.warn(f"Failed to get data for a single month: {e}")
                    continue

            if not all_data:
                return None

            combined_data = pd.concat(all_data, ignore_index=False)
            return combined_data

        except Exception as e:
            log.warn(f"Failed to get training data: {e}")
            return None

    def get_stock_pool(self, date, stock_pool):
        try:
            if stock_pool == 'HS300':
                stockList = get_index_stocks('000300.XSHG', date)
            elif stock_pool == 'ZZ500':
                stockList = get_index_stocks('399905.XSHE', date)
            elif stock_pool == 'ZZ800':
                stockList = get_index_stocks('399906.XSHE', date)
            elif stock_pool == 'CYBZ':
                stockList = get_index_stocks('399006.XSHE', date)
            elif stock_pool == 'ZXBZ':
                stockList = get_index_stocks('399101.XSHE', date)
            elif stock_pool == 'A':
                stockList = get_index_stocks('000002.XSHG', date) + get_index_stocks('399107.XSHE', date)
                stockList = [stock for stock in stockList if not stock.startswith(('68', '4', '8'))]
            elif stock_pool == 'AA':
                stockList = get_index_stocks('000985.XSHG', date)
                stockList = [stock for stock in stockList if not stock.startswith(('3', '68', '4', '8'))]
            else:
                stockList = get_index_stocks('399101.XSHE', date)

            st_data = get_extras('is_st', stockList, count=1, end_date=date)
            stockList = [stock for stock in stockList if not st_data[stock][0]]

            begin_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            filtered_stocks = []
            for stock in stockList:
                start_date = get_security_info(stock).start_date
                if start_date < (begin_date - datetime.timedelta(days=90)).date():
                    filtered_stocks.append(stock)

            return filtered_stocks[:300]

        except Exception as e:
            log.warn(f"Failed to get stock pool: {e}")
            return []

    def feature_selection(self, df):
        try:
            if 'label' not in df.columns:
                return self.jqfactors_list[:30]

            missing_counts = df[self.jqfactors_list].isnull().sum().to_dict()

            corr_matrix = df[self.jqfactors_list].corr()

            graph = {}
            threshold = 0.6

            n = len(self.jqfactors_list)
            for i in range(n):
                for j in range(i + 1, n):
                    col1, col2 = self.jqfactors_list[i], self.jqfactors_list[j]
                    corr_value = corr_matrix.iloc[i, j]

                    if not pd.isna(corr_value) and abs(corr_value) > threshold:
                        if col1 not in graph:
                            graph[col1] = []
                        graph[col1].append(col2)

                        if col2 not in graph:
                            graph[col2] = []
                        graph[col2].append(col1)

            visited = set()
            components = []

            def dfs(node, comp):
                visited.add(node)
                comp.append(node)
                if node in graph:
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            dfs(neighbor, comp)

            for col in self.jqfactors_list:
                if col not in visited:
                    comp = []
                    dfs(col, comp)
                    components.append(comp)

            to_keep = []

            for comp in components:
                if len(comp) == 1:
                    to_keep.append(comp[0])
                else:
                    # Sort by missing value count (ascending), then by name
                    comp_sorted = sorted(comp, key=lambda x: (missing_counts[x], x))
                    keep_feature = comp_sorted[0]
                    to_keep.append(keep_feature)

            log.info(f"Feature selection finished, kept {len(to_keep)} features")
            return to_keep

        except Exception as e:
            log.warn(f"Feature selection failed: {e}")
            return self.jqfactors_list[:30]


# ================= Strategy main functions and trading logic =================

def initialize(context):
    set_benchmark('399101.XSHE')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0))
    set_order_cost(
        OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003,
                  close_commission=0.0003, close_today_commission=0, min_commission=5),
        type='stock'
    )

    log.set_level('order', 'error')

    g.stock_num = 5
    g.hold_list = []
    g.trade_day_count = 0
    g.trade_interval = 10
    g.monthly_strategy = MonthlyRolling5ModelStrategy()

    # Improved model loading logic
    if not g.monthly_strategy.load_models():
        log.info("No pretrained model found, will perform first training on the first trading day")
        g.monthly_strategy.last_train_date = None
        g.need_initial_training = True  # Mark initial training needed
    else:
        log.info("5-model ensemble system (MSE-weighted) loaded successfully")
        g.need_initial_training = False  # Model already exists, no initial training needed

    g.manual_save = g.monthly_strategy.manual_save_models
    log.info("Manual save bound: use g.manual_save() to save models")

    log.info("=" * 50)
    log.info("5-model monthly rolling training strategy (MSE-weighted) initialization finished")
    log.info("=" * 50)

    run_daily(prepare_stock_list, '9:05')
    run_daily(trading_logic, '9:30')
    run_daily(check_monthly_retraining, '9:00')


def check_monthly_retraining(context):
    """Check whether monthly retraining is needed - supports initial training"""
    # If this is the first trading day and initial training is needed, train immediately
    if hasattr(g, 'need_initial_training') and g.need_initial_training:
        log.info("Perform initial model training...")
        success = g.monthly_strategy.monthly_retrain(context)
        if success:
            g.need_initial_training = False
            log.info("✅ Initial model training finished")
        else:
            log.warn("Initial model training failed, will retry on later trading days")
        return

    # Original 60-trading-day retraining logic
    if g.monthly_strategy.first_trade_date is None:
        g.monthly_strategy.first_trade_date = context.current_dt.date()
        log.info(f"Strategy started, first trading day: {g.monthly_strategy.first_trade_date}")
        return

    trade_days = get_trade_days(
        start_date=g.monthly_strategy.first_trade_date,
        end_date=context.current_dt.date()
    )
    trade_day_count = len(trade_days)

    if trade_day_count % 60 == 0:
        log.info(
            f"Total {trade_day_count} trading days accumulated, "
            f"trigger 5-model retraining (MSE-weighted)"
        )
        g.monthly_strategy.monthly_retrain(context)


def prepare_stock_list(context):
    g.hold_list = []
    for position in context.portfolio.positions.values():
        g.hold_list.append(position.security)


def trading_logic(context):
    g.trade_day_count += 1

    # Before trading, check whether model has been trained; if not, try training it
    if not g.monthly_strategy.is_trained and hasattr(g, 'need_initial_training') and g.need_initial_training:
        log.info("Model not trained, try to perform initial training...")
        success = g.monthly_strategy.monthly_retrain(context)
        if success:
            g.need_initial_training = False
            log.info("✅ Initial training finished, continue trading logic")
        else:
            log.warn("Initial training failed, fallback to baseline stock selection")

    if g.trade_day_count == 1 or g.trade_day_count % g.trade_interval == 0:
        log.info("=" * 50)
        if g.trade_day_count == 1:
            log.info("First trading day, start initial trading logic")
        else:
            log.info(f"{g.trade_day_count} trading days accumulated, execute periodic rebalancing")
        log.info("=" * 50)

        target_list = get_stock_list(context)

        if not target_list:
            log.info("No target stocks, skip trading")
            return

        log.info(f"Number of target stocks: {len(target_list)}")

        sell_count = 0
        for stock in list(context.portfolio.positions.keys()):
            if stock not in target_list:
                position = context.portfolio.positions.get(stock)
                if position and position.total_amount > 0:
                    order_target_value(stock, 0)
                    if stock in g.hold_list:
                        g.hold_list.remove(stock)
                    sell_count += 1
                    log.info(f"Sell: {stock}")

        if sell_count > 0:
            log.info(f"Selling finished: {sell_count} stocks")

        position_count = len(context.portfolio.positions)
        target_num = min(g.stock_num, len(target_list))

        if target_num > position_count and context.portfolio.cash > 0:
            cash_per_stock = context.portfolio.cash / (target_num - position_count)
            bought_count = 0

            for stock in target_list:
                if stock not in context.portfolio.positions:
                    order_target_value(stock, cash_per_stock)
                    g.hold_list.append(stock)
                    bought_count += 1
                    log.info(f"Buy: {stock}")
                    if len(context.portfolio.positions) >= target_num:
                        break

            if bought_count > 0:
                log.info(f"Buying finished: {bought_count} stocks")

        log.info(f"Trading finished, current position count: {len(context.portfolio.positions)}")


def get_stock_list(context):
    try:
        stocks = get_index_stocks('399101.XSHE', context.previous_date)
        stocks = filter_stocks(context, stocks)

        if not stocks or not hasattr(g.monthly_strategy, 'is_trained') or not g.monthly_strategy.is_trained:
            return stocks[:g.stock_num]

        factor_data = get_factor_values(
            stocks, g.monthly_strategy.selected_features,
            end_date=context.previous_date, count=1
        )
        df = pd.DataFrame(index=stocks)

        for factor in g.monthly_strategy.selected_features:
            if factor in factor_data:
                df[factor] = factor_data[factor].iloc[0, :]

        df = df.fillna(0)

        predictions = g.monthly_strategy.predict(df)
        df['score'] = predictions

        top_stocks = df.nlargest(g.stock_num * 2, 'score').index.tolist()

        log.info("Top 5 stocks prediction scores:")
        for i, stock in enumerate(top_stocks[:5], 1):
            score = df.loc[stock, 'score']
            log.info(f"{i}. {stock} - {score:.4f}")

        return top_stocks[:g.stock_num]

    except Exception as e:
        log.warn(f"Stock selection failed: {e}")
        stocks = get_index_stocks('399101.XSHE', context.previous_date)
        return stocks[:g.stock_num]


def filter_stocks(context, stocks):
    try:
        # Filter out ChiNext, STAR board, etc.
        stocks = [s for s in stocks if not s.startswith(('68', '4', '8', '3'))]

        current_data = get_current_data()
        stocks = [s for s in stocks if not getattr(current_data[s], 'is_st', False)]
        stocks = [s for s in stocks if not getattr(current_data[s], 'paused', False)]

        return stocks

    except Exception as e:
        log.warn(f"Stock filtering failed: {e}")
        return []


def manual_save_model():
    if not hasattr(g, 'monthly_strategy') or not g.monthly_strategy.is_trained:
        log.error("Model not initialized or not trained, cannot save")
        return False

    try:
        log.info("=" * 60)
        log.info("Manual save for 5-model ensemble system (MSE-weighted)")
        log.info("=" * 60)

        success = g.monthly_strategy.manual_save_models()
        if success:
            log.info("✅ Manual save succeeded")
        else:
            log.error("❌ Manual save failed")
        return success

    except Exception as e:
        log.error(f"Exception during manual save: {str(e)}")
        return False
