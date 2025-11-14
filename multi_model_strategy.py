import pandas as pd
import numpy as np
import pickle, json
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Import our factor module
from factors import get_trade_days, get_index_stocks, get_stock_pool, get_factor_values

class MonthlyRolling5ModelStrategy:
    """5-model Monthly Rolling Training Strategy (MSE-weighted ensemble)"""
    def __init__(self):
        # Factor list definition (remain same as original for reference)
        self.jqfactors_list = [
            # (list of factor names as in original code, omitted for brevity)
            'asset_impairment_loss_ttm','cash_flow_to_price_ratio','market_cap',
            'interest_free_current_liability','EBITDA','financial_assets','gross_profit_ttm',
            'net_working_capital','non_recurring_gain_loss','EBIT','sales_to_price_ratio',
            'AR','ARBR','ATR6','DAVOL10','MAWVAD','TVMA6','PSY','VOL10','VDIFF','VEMA26',
            'VMACD','VOL120','VOSC','VR','WVAD','arron_down_25','arron_up_25','BBIC',
            'MASS','Rank1M','single_day_VPT','single_day_VPT_12','single_day_VPT_6','Volume1M',
            'capital_reserve_fund_per_share','net_asset_per_share','net_operate_cash_flow_per_share',
            'operating_profit_per_share','total_operating_revenue_per_share','surplus_reserve_fund_per_share',
            'ACCA','account_receivable_turnover_days','account_receivable_turnover_rate',
            'adjusted_profit_to_total_profit','super_quick_ratio','MLEV','debt_to_equity_ratio',
            'debt_to_tangible_equity_ratio','equity_to_fixed_asset_ratio','fixed_asset_ratio',
            'intangible_asset_ratio','invest_income_associates_to_total_profit','long_debt_to_asset_ratio',
            'long_debt_to_working_capital_ratio','net_operate_cash_flow_to_total_liability',
            'net_operating_cash_flow_coverage','non_current_asset_ratio','operating_profit_to_total_profit',
            'roa_ttm','roe_ttm','Kurtosis120','Kurtosis20','Kurtosis60','sharpe_ratio_20',
            'sharpe_ratio_60','Skewness120','Skewness20','Skewness60','Variance120','Variance20',
            'liquidity','beta','book_to_price_ratio','cash_earnings_to_price_ratio','cube_of_size',
            'earnings_to_price_ratio','earnings_yield','growth','momentum','natural_log_of_market_cap',
            'boll_down','MFI14','MAC10','fifty_two_week_close_rank','price_no_fq'
        ]
        # Model configurations
        self.models_config = {
            'lgb': {'name': 'LightGBM', 'model': None, 'mse': None, 'weight': None},
            'xgb': {'name': 'XGBoost', 'model': None, 'mse': None, 'weight': None},
            'svr': {'name': 'SVR', 'model': None, 'scaler': None, 'mse': None, 'weight': None},
            'rf': {'name': 'RandomForest', 'model': None, 'mse': None, 'weight': None},
            'lr': {'name': 'LinearRegression', 'model': None, 'mse': None, 'weight': None}
        }
        self.selected_features = []
        self.last_train_date = None
        self.is_trained = False
        self.feature_scaler = None
        self.training_months = 36
        self.initial_model_loaded = False
        self.first_trade_date = None
        # File paths for saved model and metadata
        self.model_filepath = 'models/5model_monthly_ensemble_mse.pkl'
        self.metadata_filepath = 'models/5model_metadata_mse.json'
        # Training history
        self.training_losses = []
        self.training_auc_scores = []
        self.training_model_mses = []
        self.episode_count = 0

    def monthly_retrain(self, current_date, stock_pool='ZXBZ'):
        """Perform monthly retraining of models (with MSE-based dynamic weighting)."""
        try:
            # If models were just loaded at start, skip first retrain as per original logic
            if self.initial_model_loaded:
                print("Initial model loaded, skipping first 60-day retraining")
                self.initial_model_loaded = False
                return False
            # Check if already trained this month
            if self.last_train_date and current_date.month == self.last_train_date.month and current_date.year == self.last_train_date.year:
                print("Already trained this month, skipping retrain")
                return True
            print("▶ Starting monthly retraining for 5-model ensemble...")
            # Prepare training data range: last training_months months up to previous day
            end_date = (current_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=self.training_months)).strftime('%Y-%m-%d')
            train_data = self.get_training_data(start_date, end_date, stock_pool)
            if train_data is None or len(train_data) < 100:
                print("WARNING: Not enough training data, using existing models")
                return False
            # Feature selection (if needed, otherwise use default top factors or prior selected)
            selected_features = self.feature_selection(train_data)
            if len(selected_features) < 10:
                print("WARNING: Too few effective features, using previously selected features")
                selected_features = self.selected_features if self.selected_features else self.jqfactors_list[:30]
            print(f"Selected {len(selected_features)} features for training.")
            # Prepare feature matrix and labels
            X = train_data[selected_features].fillna(0)  # fill NA if any (our factors should minimize NA)
            y = train_data['label']
            # Robust scaling
            X_processed = self.robust_data_preprocessing(X)
            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
            print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
            # Train models
            success_count = 0
            model_mses = {}
            if self.train_lightgbm(X_train, y_train, X_val, y_val, model_mses): success_count += 1
            if self.train_xgboost(X_train, y_train, X_val, y_val, model_mses): success_count += 1
            if self.train_svr(X_train, y_train, X_val, y_val, model_mses): success_count += 1
            if self.train_random_forest(X_train, y_train, X_val, y_val, model_mses): success_count += 1
            if self.train_linear_regression(X_train, y_train, X_val, y_val, model_mses): success_count += 1
            if success_count >= 3:
                # Save MSEs and calculate weights
                for model_name, mse in model_mses.items():
                    self.models_config[model_name]['mse'] = mse
                self.calculate_dynamic_weights()
                self.selected_features = selected_features
                self.is_trained = True
                self.last_train_date = current_date
                self.episode_count += 1
                self.training_model_mses.append(model_mses)
                # Evaluate ensemble on validation
                auc_score = self.evaluate_ensemble(X_val, y_val)
                self.training_auc_scores.append(auc_score)
                # Save model to disk
                save_success = self.save_models(selected_features, auc_score)
                if save_success:
                    print("✅ Model saved successfully.")
                else:
                    print("❌ Model save failed!")
                print(f"✅ Retraining complete! Models trained: {success_count}/5, Ensemble AUC: {auc_score:.4f}")
                return True
            else:
                print("WARNING: Fewer than 3 models trained successfully, keeping existing models.")
                return False
        except Exception as e:
            print(f"Retraining failed: {e}")
            return False

    def get_training_data(self, start_date, end_date, stock_pool='ZXBZ'):
        """Construct labeled training dataset from historical period [start_date, end_date]."""
        try:
            # Get all trade dates in range
            all_dates = get_trade_days(start_date, end_date)
            # Choose approximately 2 dates per month (1st and 15th trading days) as sampling points
            monthly_dates = [pd.to_datetime(d) for d in all_dates if pd.to_datetime(d).day in (1,15)]
            if len(monthly_dates) < 2:
                return None
            all_data = []
            for i in range(len(monthly_dates)-1):
                date = monthly_dates[i]
                next_date = monthly_dates[i+1]
                date_str = date.strftime('%Y-%m-%d')
                next_date_str = next_date.strftime('%Y-%m-%d')
                # Get stock pool constituents on this date
                stocks = get_stock_pool(date_str, stock_pool)
                if not stocks:
                    continue
                # Get factor values for all stocks on this date
                df = get_factor_values(stocks, date_str)
                if df.empty:
                    continue
                # Drop stocks with any missing factor (to ensure robust training data)
                df = df.dropna(axis=0)
                if df.empty:
                    continue
                # Compute forward return between date and next_date for each stock
                price_df = get_price_df(df.index.tolist(), date_str, next_date_str)
                # Determine label by median forward return
                returns = []
                valid_stocks = []
                for stock in df.index:
                    if stock not in price_df.index or price_df.loc[stock].dropna().shape[0] < 2:
                        # If we don't have at least two price points (stock didn't trade throughout period), skip
                        continue
                    # forward return from first to last available trading day in [date, next_date]
                    prices = price_df.loc[stock].dropna()
                    start_price = prices.iloc[0]
                    end_price = prices.iloc[-1]
                    if start_price is None or end_price is None:
                        continue
                    ret = end_price / start_price - 1.0
                    valid_stocks.append(stock)
                    returns.append(ret)
                if len(valid_stocks) < 10:
                    continue
                df = df.loc[valid_stocks]
                df['pchg'] = returns
                # Label as 1 if return >= median return of this period, else 0
                median_ret = df['pchg'].median()
                df['label'] = (df['pchg'] >= median_ret).astype(int)
                df.drop(columns=['pchg'], inplace=True)
                all_data.append(df)
            if not all_data:
                return None
            combined_data = pd.concat(all_data, ignore_index=False)
            return combined_data
        except Exception as e:
            print(f"Error constructing training data: {e}")
            return None

    def calculate_dynamic_weights(self):
        """Calculate ensemble weights for each model proportional to 1/MSE."""
        valid_models = [(m, cfg['mse']) for m,cfg in self.models_config.items() if cfg['model'] is not None and cfg['mse'] is not None]
        if not valid_models:
            print("❌ No valid models to weight.")
            return
        epsilon = 1e-8
        inv_mses = [1/(mse + epsilon) for _, mse in valid_models]
        total_inv = sum(inv_mses)
        for (model_name, _), inv in zip(valid_models, inv_mses):
            weight = inv/total_inv
            self.models_config[model_name]['weight'] = weight
            print(f"{self.models_config[model_name]['name']} weight: {weight:.4f} (MSE: {self.models_config[model_name]['mse']:.6f})")

    def robust_data_preprocessing(self, X):
        """Robust data preprocessing (handle NA/inf and scale features)."""
        try:
            X_proc = X.copy().replace([np.inf, -np.inf], np.nan)
            for col in X_proc.columns:
                if X_proc[col].isnull().any():
                    median_val = X_proc[col].median()
                    X_proc[col].fillna(median_val, inplace=True)
            if self.feature_scaler is None:
                self.feature_scaler = RobustScaler()
                X_scaled = self.feature_scaler.fit_transform(X_proc)
            else:
                X_scaled = self.feature_scaler.transform(X_proc)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip extreme values
            X_scaled = np.clip(X_scaled, -10, 10)
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        except Exception:
            return X.fillna(0).replace([np.inf, -np.inf], 0)

    # Model training functions (each computes MSE on validation and stores model)
    def train_lightgbm(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 20,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
                'verbosity': -1, 'random_state': 42
            }
            model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_val],
                               callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            model_mses['lgb'] = mse
            print(f"LightGBM trained, MSE: {mse:.6f}")
            self.models_config['lgb']['model'] = model
            return True
        except Exception as e:
            print(f"LightGBM training failed: {e}")
            return False

    def train_xgboost(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            params = {
                'objective': 'binary:logistic', 'learning_rate': 0.05, 'max_depth': 6,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                'n_estimators': 300, 'use_label_encoder': False
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict_proba(X_val)[:, 1]
            mse = mean_squared_error(y_val, y_pred)
            model_mses['xgb'] = mse
            print(f"XGBoost trained, MSE: {mse:.6f}")
            self.models_config['xgb']['model'] = model
            return True
        except Exception as e:
            print(f"XGBoost training failed: {e}")
            return False

    def train_svr(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            # To speed up, use a sample if training set is large
            sample_size = min(5000, len(X_train_scaled))
            if sample_size < len(X_train_scaled):
                idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
                X_train_sampled = X_train_scaled[idx]
                y_train_sampled = y_train.iloc[idx]
            else:
                X_train_sampled = X_train_scaled
                y_train_sampled = y_train
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(X_train_sampled, y_train_sampled)
            y_pred = model.predict(X_val_scaled)
            # Normalize SVR outputs to [0,1] range for comparison
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
            mse = mean_squared_error(y_val, y_pred_norm)
            model_mses['svr'] = mse
            print(f"SVR trained, MSE: {mse:.6f}")
            self.models_config['svr']['model'] = model
            self.models_config['svr']['scaler'] = scaler
            return True
        except Exception as e:
            print(f"SVR training failed: {e}")
            return False

    def train_random_forest(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=20,
                                          random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
            mse = mean_squared_error(y_val, y_pred_norm)
            model_mses['rf'] = mse
            print(f"RandomForest trained, MSE: {mse:.6f}")
            self.models_config['rf']['model'] = model
            return True
        except Exception as e:
            print(f"Random Forest training failed: {e}")
            return False

    def train_linear_regression(self, X_train, y_train, X_val, y_val, model_mses):
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
            mse = mean_squared_error(y_val, y_pred_norm)
            model_mses['lr'] = mse
            print(f"LinearRegression trained, MSE: {mse:.6f}")
            self.models_config['lr']['model'] = model
            return True
        except Exception as e:
            print(f"Linear regression training failed: {e}")
            return False

    def evaluate_ensemble(self, X_val, y_val):
        """Evaluate ensemble model on validation data using current weights, return AUC."""
        try:
            preds = []
            weights = []
            used_models = []
            for m_name, cfg in self.models_config.items():
                if cfg['model'] is None or cfg.get('weight') is None:
                    continue
                # Get prediction for validation set
                if m_name == 'lgb':
                    pred = cfg['model'].predict(X_val)
                elif m_name == 'xgb':
                    pred = cfg['model'].predict_proba(X_val)[:, 1] if hasattr(cfg['model'], 'predict_proba') else cfg['model'].predict(X_val)
                elif m_name == 'svr':
                    X_val_scaled = cfg['scaler'].transform(X_val)
                    pred_raw = cfg['model'].predict(X_val_scaled)
                    # normalize SVR output
                    pred = (pred_raw - pred_raw.min())/(pred_raw.max()-pred_raw.min()+1e-8)
                elif m_name == 'rf':
                    pred_raw = cfg['model'].predict(X_val)
                    pred = (pred_raw - pred_raw.min())/(pred_raw.max()-pred_raw.min()+1e-8)
                elif m_name == 'lr':
                    pred_raw = cfg['model'].predict(X_val)
                    pred = (pred_raw - pred_raw.min())/(pred_raw.max()-pred_raw.min()+1e-8)
                else:
                    continue
                preds.append(pred)
                weights.append(cfg['weight'])
                used_models.append(cfg['name'])
            if not preds:
                return 0.5
            ensemble_pred = np.average(np.vstack(preds), axis=0, weights=weights)
            auc = roc_auc_score(y_val, ensemble_pred)
            print(f"Ensemble evaluation – models used: {used_models}, weights: {[round(w,4) for w in weights]}, AUC: {auc:.4f}")
            return auc
        except Exception as e:
            print(f"Ensemble evaluation failed: {e}")
            return 0.5

    def predict(self, X):
        """Predict ensemble scores for given feature DataFrame X."""
        try:
            if not self.is_trained or not self.selected_features:
                print("Model not trained, returning random predictions")
                return np.random.random(len(X))
            X_proc = self.robust_data_preprocessing(X[self.selected_features])
            preds = []
            weights = []
            for m_name, cfg in self.models_config.items():
                if cfg['model'] is None or cfg.get('weight') is None:
                    continue
                if m_name == 'lgb':
                    pred = cfg['model'].predict(X_proc)
                elif m_name == 'xgb':
                    pred = cfg['model'].predict_proba(X_proc)[:, 1] if hasattr(cfg['model'], 'predict_proba') else cfg['model'].predict(X_proc)
                elif m_name == 'svr':
                    pred_raw = cfg['model'].predict(cfg['scaler'].transform(X_proc))
                    pred = (pred_raw - pred_raw.min())/(pred_raw.max()-pred_raw.min()+1e-8)
                elif m_name == 'rf':
                    pred_raw = cfg['model'].predict(X_proc)
                    pred = (pred_raw - pred_raw.min())/(pred_raw.max()-pred_raw.min()+1e-8)
                elif m_name == 'lr':
                    pred_raw = cfg['model'].predict(X_proc)
                    pred = (pred_raw - pred_raw.min())/(pred_raw.max()-pred_raw.min()+1e-8)
                else:
                    continue
                preds.append(pred)
                weights.append(cfg['weight'])
            if not preds:
                return np.zeros(len(X))
            ensemble_scores = np.average(np.vstack(preds), axis=0, weights=weights)
            return ensemble_scores
        except Exception as e:
            print(f"Prediction failed: {e}")
            return np.random.random(len(X))

    def save_models(self, selected_features, auc_score, filepath=None):
        """Save models and metadata to file."""
        try:
            if filepath is None:
                filepath = self.model_filepath
            model_data = {
                'models_config': self.models_config,
                'selected_features': selected_features,
                'feature_scaler': self.feature_scaler,
                'auc_score': auc_score,
                'saved_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'training_losses': self.training_losses,
                'training_auc_scores': self.training_auc_scores,
                'training_model_mses': self.training_model_mses,
                'episode_count': self.episode_count,
                'last_train_date': self.last_train_date.strftime('%Y-%m-%d') if self.last_train_date else None
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            # Save metadata JSON (for quick view of training history)
            metadata = {
                'training_losses': self.training_losses,
                'training_auc_scores': self.training_auc_scores,
                'training_model_mses': self.training_model_mses,
                'episode_count': self.episode_count,
                'last_train_date': self.last_train_date.strftime('%Y-%m-%d') if self.last_train_date else None,
                'save_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'selected_features_count': len(selected_features),
                'current_auc': auc_score
            }
            with open(self.metadata_filepath, 'w') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Model save failed: {e}")
            return False

    def load_models(self, filepath=None):
        """Load models from file (including MSE and weights)."""
        try:
            if filepath is None:
                filepath = self.model_filepath
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.models_config = model_data['models_config']
            self.selected_features = model_data['selected_features']
            self.feature_scaler = model_data.get('feature_scaler')
            self.is_trained = True
            self.training_losses = model_data.get('training_losses', [])
            self.training_auc_scores = model_data.get('training_auc_scores', [])
            self.training_model_mses = model_data.get('training_model_mses', [])
            self.episode_count = model_data.get('episode_count', 0)
            last_date_str = model_data.get('last_train_date')
            if last_date_str:
                self.last_train_date = pd.to_datetime(last_date_str)
            self.initial_model_loaded = True
            print(f"✅ Loaded saved model ensemble from {filepath}")
            print(f"Features: {len(self.selected_features)}, past trainings: {self.episode_count}")
            return True
        except Exception as e:
            print(f"Model load failed: {e}")
            return False

    def manual_save_models(self):
        """Manually trigger model save (for use mid-backtest if needed)."""
        if not self.is_trained:
            print("No trained model to save.")
            return False
        try:
            print("="*60)
            print("Manual save of 5-model ensemble")
            print("="*60)
            current_auc = self.training_auc_scores[-1] if self.training_auc_scores else 0.5
            success = self.save_models(self.selected_features, current_auc)
            if success:
                print("✅ Manual save successful.")
            else:
                print("❌ Manual save failed.")
            return success
        except Exception as e:
            print(f"Manual save error: {e}")
            return False

    def feature_selection(self, df):
        """Feature selection: remove highly collinear factors and those with excessive missing values."""
        try:
            if 'label' not in df.columns:
                return self.jqfactors_list[:30]
            # Remove features with too many missing values
            # (In our data, we already dropped NAs, so this might not be needed)
            missing = df.isnull().sum()
            # Compute correlation matrix for feature de-collinearity
            corr_matrix = df.drop(columns=['label']).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find features with correlation > 0.6
            to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]
            selected = [col for col in df.columns if col not in to_drop and col != 'label']
            print(f"Feature selection dropped {len(to_drop)} highly correlated features, kept {len(selected)}.")
            return selected
        except Exception as e:
            print(f"Feature selection error: {e}")
            # Fallback: use first 30 factors
            return self.jqfactors_list[:30]

# Example usage (outside of this module, e.g., in backtesting loop):
# strategy = MonthlyRolling5ModelStrategy()
# # Load existing model if available
# strategy.load_models()
# # Each day in backtest:
# context_date = current_date  # assume current_date is a datetime
# # Periodically retrain (e.g., every 60 trading days or at month start)
# strategy.monthly_retrain(context_date, stock_pool='ZXBZ')
# # Select top stocks to trade:
# stock_pool = get_stock_pool(context_date.strftime('%Y-%m-%d'), 'ZXBZ')
# if stock_pool:
#     factor_df_today = get_factor_values(stock_pool, context_date.strftime('%Y-%m-%d'))
#     scores = strategy.predict(factor_df_today)
#     factor_df_today['score'] = scores
#     top_stocks = factor_df_today.nlargest(5, 'score').index.tolist()
#     print("Top 5 stocks:", top_stocks)
