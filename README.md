Machine-Learning Multi-Strategy: 5-Model MSE-Weighted Rolling Training (JoinQuant Only)

Important
This strategy is written specifically for the JoinQuant (聚宽) research platform and depends on JoinQuant’s built-in APIs (jqdata, jqfactor, g, context, run_daily, etc.).
It cannot run as-is in a pure local Python environment.
Any offline / local deployment requires access to JoinQuant’s official research environment (formal/paid account) and additional integration work, which is not included in this repository.

1. Overview

This project implements a multi-model, dynamically weighted ensemble stock-picking system on JoinQuant. The core idea:

Train 5 different machine-learning models on a rolling window:

LightGBM

XGBoost

SVR

Random Forest

Linear Regression

Evaluate each model on a validation set using MSE (mean squared error).

Compute dynamic ensemble weights based on the (rolling) MSE of each model:

Models with lower error get higher weight.

Models that consistently underperform are penalized.

Rebalance the portfolio every 10 trading days, holding a small basket of high-scoring stocks (e.g. 5 names) with simple risk filters and diversification.

This strategy is intentionally “framework-oriented” rather than fully optimized.
The focus is on:

how to design a rolling training pipeline in JoinQuant,

how to combine multiple models via dynamic weights,

and how to plug it into a simple, periodic rebalancing workflow.

2. Platform & Environment Requirements (JoinQuant Only)

This code:

Assumes it is running inside the JoinQuant research environment, with:

from jqdata import *

from jqfactor import get_factor_values

access to JoinQuant’s data (prices, factors, index constituents, trading calendar, etc.)

Relies on JoinQuant’s strategy lifecycle:

initialize(context)

run_daily(...)

g (global state), context (trading context)

order_target_value, set_benchmark, set_slippage, set_order_cost, etc.

Out of the box, this repository does not provide:

Any local CSV/database data loader

Any standalone backtest engine

Any replacement for JoinQuant’s internal APIs

Local / offline usage
To run something similar locally, you would need:

an official JoinQuant account with access to their formal research environment / APIs, and

your own implementation of data providers + a backtest loop.
That integration is outside the scope of this repository.

3. High-Level Strategy Flow

At a high level, the strategy works like this:

Initialization (initialize)

Set benchmark to the Shenzhen Composite Index: 399101.XSHE.

Configure slippage and transaction costs (commission ~0.03%, stamp duty ~0.1%, etc.).

Instantiate MonthlyRolling5ModelStrategy (the “core” class).

Try to load a previously saved model from disk.

If load succeeds: mark the system as “pretrained”.

If load fails: mark as “untrained”, and the strategy will perform first training during backtest.

Register daily callbacks:

check_monthly_retraining (09:00)

prepare_stock_list (09:05)

trading_logic (09:30)

Rolling Training (every ~quarter)

Use historical data from the past 36 months as the training set.

Build factor features and labels.

Train 5 models, compute MSE on validation data.

Compute dynamic weights based on MSE (with rolling average + penalty).

Save models and metadata to file in the JoinQuant research environment.

Daily Stock Scoring

Build a filtered stock universe from an index (e.g. Shenzhen Composite) with risk filters.

Pull factor values for candidate stocks.

Run each model to get scores, then combine them using dynamic weights.

Periodic Rebalancing (every 10 trading days)

Sell positions that are no longer in the target list.

Buy high-score stocks to maintain e.g. 5 equally weighted holdings.

Simple liquidity and risk checks.

4. Model Training & Dynamic Weighting
4.1 Rolling Training Logic

The strategy uses a rolling training mechanism (monthly_retrain):

Trigger frequency

After an initial warm-up, retraining is triggered every fixed interval in trading days (e.g. every 60 trading days ≈ 3 months).

The interval is configurable; you can adjust it to your own research design.

Training data

Time window: last 36 months of history.

Universe: index constituents (e.g. 399101.XSHE), filtered by:

remove ST stocks,

remove ChiNext, STAR board, and similar,

remove newly listed stocks (e.g. listed < 90 days).

Features: ~79 factors across four categories:

Fundamental (e.g. profitability, leverage, cash flow)

Valuation (e.g. PE, book-to-price)

Technical (e.g. ATR, MACD, momentum, volume/volatility indicators)

Risk (e.g. beta, return variance, skewness, kurtosis)

Labels:

Regression target: future return over a certain horizon.

Binary label: whether this future return is above the cross-section median (1 = above median, 0 = below).

3-class label: based on return quantiles (25% / 75%) – low / medium / high.

Data preprocessing

Fill missing values with median per feature.

Replace inf / -inf with NaN and clean.

Clip extreme values into a safe range (e.g. [-10, 10]).

Use RobustScaler to standardize features (more robust to outliers).

Feature selection

Compute the correlation matrix for all candidate factors.

Build a graph of high-correlation pairs (|corr| > 0.6).

Within each correlated group, keep only one factor, preferring:

lower missing ratio,

stable naming, etc.

This reduces multicollinearity and typically keeps ≈ 30 core features.

4.2 5-Model Ensemble

The following models are trained on the same feature space:

LightGBM – tree boosting, good at non-linear patterns

XGBoost – another tree boosting variant, robust and battle-tested

SVR – kernel method capturing high-dimensional decision boundaries

Random Forest – ensemble of trees, decent robustness and stability

Linear Regression – simple baseline, low variance, stabilizes the ensemble

Each model is trained using the binary label (>= median return) and evaluated on a validation set using MSE between predicted probability and true label.

4.3 Dynamic Weighting (MSE with Rolling Window & Penalty)

Naively using single-run MSE for weights leads to unstable ensemble behavior, because one noisy validation split can dominate the weight allocation.

To address this, the strategy uses:

Rolling MSE window

For each model, keep the MSE values of the last 3 training rounds.

Define the model’s "rolling MSE" as the average over this window.

Low-contribution penalty

Compute the global mean of rolling MSE among valid models.

If a model’s last 2 rolling MSE values are both greater than 1.5 × global_mean_mse, treat it as a “low-contribution” model and apply a penalty (e.g. multiply weight by 0.5).

Weight formula

For each model i, define:

rolling_mse_i: rolling average MSE over the last 3 rounds

penalty_i:

1.0 for normal models

0.5 for low-contribution models

ε: a small constant (e.g. 1×10−8) to avoid division by zero

Then the weight of model i is:

weight_i = (1/(rolling_mse_i + ε) * penalty_i) / Σk=1..n [1/(rolling_mse_k + ε) * penalty_k]

Lower MSE → larger 1/rolling_mse_i → larger weight

Consistently bad models get explicitly penalized via penalty_i = 0.5.

The final prediction is a weighted average of all models’ scores using these dynamic weights.

5. Stock Selection Logic

Stock selection is handled primarily by get_stock_list:

Base Universe

Start from index constituents (e.g. Shenzhen Composite 399101.XSHE).

Risk Filters

Remove:

ChiNext / STAR / BSE tickers (e.g. codes starting with 68, 3, 4, 8).

ST stocks.

Suspended stocks.

Newly listed stocks (< 90 days).

Factor Data & Scoring

Fetch factor values for the remaining stocks using get_factor_values (JoinQuant).

Use the trained 5-model ensemble to score each stock.

Higher score ≈ higher predicted probability of “high future return”.

Final Target List

Sort by score in descending order.

Take the top N as target portfolio (e.g. top 5 stocks).

If the model has not been trained yet (e.g. very first run), the strategy falls back to a simple baseline: select the first few stocks in the filtered universe to avoid stalling.

6. Trading & Rebalancing Logic

Actual trading is executed in trading_logic:

Rebalance frequency

First trading day: always rebalance to build an initial portfolio.

Afterwards: rebalance every 10 trading days.

Sell rules

For each current position:

If the stock is not in the latest target list → sell (target position = 0).

Optional: special handling can be added for limit-up situations, etc.

Buy rules

Compute the number of stocks to hold (e.g. 5).

Allocate available cash equally among new positions.

For each stock in the target list:

If not already held → buy up to the target value.

Basic liquidity/risk checks can be applied (e.g. avoid limit-up/limit-down names).

Risk control (basic)

Fixed number of holdings → natural diversification.

Simple index benchmark for relative performance tracking.

Trading frequency capped at every 10 trading days to reduce turnover and costs.

The trading part is intentionally simple and “rough”; the focus of this repo is the rolling training + dynamic ensemble framework.

7. Design Goals & Limitations

Goals

Demonstrate a multi-model ensemble framework with:

Rolling training window,

Dynamic weight allocation based on MSE,

Penalty for chronically underperforming models.

Show how to integrate machine learning models into a JoinQuant stock-picking strategy.

Provide a research template that can be extended with:

Better factor engineering,

Risk models,

More advanced execution logic.

Limitations

The current parameter choices (training window length, rebalance period, stock count, etc.) are for demonstration only.

The strategy is highly dependent on:

Market regime,

Data period,

Factor quality.

The trading logic does not distinguish detailed intraday timing or advanced execution constraints.

The code is tightly coupled with JoinQuant and is not plug-and-play for generic local Python environments.

This repository is intended for research and education, not for live trading or investment advice.

8. How to Use (on JoinQuant)

Log in to your JoinQuant research environment.

Create a new strategy or research script.

Copy the strategy code into that script.

Make sure your account has permission to:

Access the required data (jqdata, jqfactor),

Read/write local files for model persistence.

Run a backtest:

If there is no saved model file, the strategy will:

Train automatically on the first run,

Save models and metadata for future reuse.

If a saved model exists and is loaded, rolling retraining will start after the configured warm-up period and then recur at fixed intervals.

9. Roadmap / Possible Extensions

If you want to extend this framework, typical next steps include:

Factor Engineering

Add or refine factor sets (industry-neutral factors, style factors, alternative data).

Normalize / standardize in cross-section, add industry neutralization, etc.

Model Side

Add other models (e.g. CatBoost, neural networks).

Implement cross-validation and more robust hyperparameter tuning.

Risk Management

Sector/industry diversification constraints.

Volatility or drawdown-based position sizing.

Execution

More realistic trading calendar,

Intraday price handling,

Slippage models.

Local / Offline Version

Build your own DataProvider layer,

Implement a custom backtest loop,

Integrate with JoinQuant’s official research terminal (requires formal account).
