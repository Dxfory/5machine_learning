# 🧠 Machine-Learning Multi-Strategy  
## **5-Model MSE-Weighted Rolling Training (JoinQuant Only)**

---

## ⚠️ **Important**

This strategy is written **specifically for the JoinQuant (聚宽)** research platform and depends on JoinQuant’s built-in APIs (`jqdata`, `jqfactor`, `g`, `context`, `run_daily`, etc.).

- ❌ It **cannot** run as-is in a local Python environment  
- 📝 Offline/local deployment requires:
  - A formal **JoinQuant paid research account**
  - Custom integration work (data loader + backtest engine)

---

# **1. Overview**

This project implements a **multi-model, dynamically weighted ensemble stock-picking system** on JoinQuant.

### **Core idea**
Train **5 machine-learning models** on a rolling window:

- LightGBM  
- XGBoost  
- SVR  
- Random Forest  
- Linear Regression  

Evaluate models using **MSE (mean squared error)** and compute **dynamic ensemble weights**:

- Lower error → higher weight  
- Persistently poor models → penalized  

Rebalance every **10 trading days**, holding **top 5 high-scoring stocks**, with simple risk filters.

This is a **framework-oriented** strategy emphasizing:

- Rolling training pipeline design  
- Multi-model dynamic weighting  
- JoinQuant periodic rebalancing workflow  

---

# **2. Platform & Environment Requirements (JoinQuant Only)**

This code assumes it is running **inside JoinQuant** with:

```python
from jqdata import *
from jqfactor import get_factor_values
```

Access to:

- Price data  
- Factor data  
- Index constituents  
- Trading calendar  
- Strategy lifecycle APIs (`initialize`, `run_daily`, `order_target_value`, etc.)  

### ❌ This repository does NOT include:
- Local CSV/database loader  
- Independent backtest engine  
- Replacement for JQ internal APIs  

### 🛠 Running locally requires:
- A formal JoinQuant API environment  
- A custom data provider + backtest loop  
(Outside the scope of this repository.)  

---

# **3. High-Level Strategy Flow**

### **Initialization (`initialize`)**
- Set benchmark: **399101.XSHE (Shenzhen Composite)**
- Configure slippage & transaction cost  
- Instantiate `MonthlyRolling5ModelStrategy`  
- Load or train model  

Register callbacks:

| Time | Task | Function |
|------|------|-----------|
| 09:00 | Monthly retrain check | `check_monthly_retraining` |
| 09:05 | Select universe | `prepare_stock_list` |
| 09:30 | Trading/rebalancing | `trading_logic` |

### **Rolling Training (≈ every 3 months)**
- Train on **36 months** of history  
- Compute MSE  
- Update rolling-MSE window  
- Apply penalty for weak models  
- Save model files  

### **Daily Stock Scoring**
- Filter universe  
- Fetch factor values  
- Score via weighted ensemble  

### **Periodic Rebalancing (every 10 days)**
- Sell dropped names  
- Buy/hold top N stocks  
- Basic risk filters  

---

# **4. Model Training & Dynamic Weighting**

## **4.1 Rolling Training Logic**

### Training Frequency
- Retrain every **60 trading days** (configurable)

### Training Data
- Window: **36 months**
- Universe filters:
  - Remove ST stocks  
  - Remove ChiNext/STAR/BSE boards  
  - Remove newly listed (<90 days)  

### Features (~79 factors)
Categories:
- Fundamental  
- Valuation  
- Technical  
- Risk  

### Labels
- Regression: future return  
- Binary: above/below cross-section median  
- 3-class: return quantiles (25% / 75%)  

### Preprocessing
- Median fill  
- Replace inf  
- Outlier clipping  
- RobustScaler normalization  

### Feature Selection
- Correlation graph (|corr| > 0.6)  
- Keep 1 factor per correlated cluster  

---

## **4.2 Five-Model Ensemble**

Models trained on identical features:

| Model | Advantage |
|-------|-----------|
| LightGBM | Non-linear boosting |
| XGBoost | Stable boosting variant |
| SVR | High-dimensional boundaries |
| Random Forest | Robust ensembles |
| Linear Regression | Stabilizes ensemble |

All evaluated using **MSE** on validation set.

---

## **4.3 Dynamic Weighting Formula**

### Rolling MSE Window
- Keep last **3 rounds**
- Use average as **rolling_mse**

### Low-Contribution Penalty
If recent rolling MSE > 1.5 × mean → **penalty = 0.5**  

### **Weight Formula**

<img width="1296" height="261" alt="image" src="https://github.com/user-attachments/assets/5dc0f7d1-be9a-47f4-ab5a-49481829add5" />
**Weight formula**

\[
w_i \propto \frac{1}{\mathrm{RollingLoss}_i + \epsilon} \times \mathrm{Penalty}_i
\]

where Penalty\_i = 0.5 if the last two rolling losses are both > 1.5× the global mean, otherwise 1.0.


- Lower MSE → larger weight  
- Weak models → explicit penalty  

Final score = weighted average of 5 model predictions.

---

# **5. Stock Selection Logic**

### Base Universe
- Index constituents (399101.XSHE)

### Risk Filters
Remove:
- ChiNext / STAR / BSE  
- ST stocks  
- Suspended stocks  
- Newly listed (<90 days)  

### Factor Data & Scoring
- Fetch using `get_factor_values`  
- Run ensemble to generate score  
- Higher score → better expected return  

### Final Portfolio
- Sort by score  
- Select **Top 5** stocks  
- If model untrained → fallback baseline selection  

---

# **6. Trading & Rebalancing Logic**

### Rebalance Frequency
- First day → always  
- Then every **10 trading days**  

### Sell Rules
- If stock not in target list → sell  

### Buy Rules
- Equal-weight among target holdings  
- Check liquidity/risk  

### Risk Control (Basic)
- Fixed number of holdings  
- Low turnover  
- Benchmark tracking  

Trading logic is intentionally **simple**, focusing on demonstrating **framework**, not execution.

---

# **7. Design Goals & Limitations**

## Goals
- Multi-model ensemble with rolling training  
- Dynamic weight allocation  
- Penalty for underperforming models  
- End-to-end JoinQuant ML workflow  

## Limitations
- Parameters are for demonstration  
- Sensitive to:
  - Market regime  
  - Factor quality  
  - Data period  
- Execution logic basic  
- JoinQuant-dependent → not plug-and-play  

---

# **8. How to Use (JoinQuant)**

1. Log into JoinQuant research environment  
2. Create a new strategy  
3. Copy the entire code into the script  
4. Ensure you have:
   - `jqdata` & `jqfactor` permissions  
   - Read/write local file access  

### **Backtest**
- If no saved model → first run triggers full training  
- Then rolling training automatically happens per schedule  

---

# **9. Roadmap / Possible Extensions**

### **Factor Engineering**
- Industry-neutral factors  
- Style factors  
- Alternative data  

### **Model Improvements**
- CatBoost / NN models  
- Cross-validation  
- Hyperparameter optimization  

### **Risk Management**
- Industry diversification  
- Volatility-based allocation  
- Dynamic stop-loss  

### **Execution Layer**
- More realistic slippage  
- Intraday modeling  
- Liquidity constraints  

### **Local/Offline Version**
- Custom DataProvider  
- Custom backtest loop  
- Integrate with JoinQuant official API (paid)  

---

