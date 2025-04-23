# qf-chitrea
This is a GitHub place for Alan and Dominic to work on the quantitative finance project in Chinese Treasury (qf-chitrea) for model prediction and training. 

# Current Progress (Please read/edit the note before you start)
- Run Machine Learning models in Python

# Objective
1. Quantitative model for bond
   We will use the following data: 
   - 中國_中債國債到期收益率_10年 as the target variable for prediction
   - 中債國債總淨價指數 as the performance benchmark
   - 因子signal汇总、估值因子signal、宏观因子signal、因子汇总 as the factor library
  
**Modelling Approach:
**We can start by training a base model using OLS or Ridge regression (and later extend to models like LSTM or Prophet) 

2. Insight into Chinese bond and interest rate and their prediction model in the long term
   - Build a long-term interest rate prediction model
   - Use macro variables such as CPI, PPI, M1 and Total Social Financing as features
   - Incorporate valuation factors and economic cycle-related variables for forecasting (e.g. marked as +1, 0, -1)

3. Time-series analysis
   We can use:
   - Moving averages, lag features, and trend changes
   - Trend + seasonality models built with statsmodels or Prophet
  
   Model Evaluation Metrics:
   MSE, Sharpe Ratio, Maximum Drawdown, etc

# Project Structure
1. Data Preparation (DONE)
   - Standardize variables
   - Handling missing values
   - Reduce dimensionality (e.g. PCA, correlation filter)
   - Create lag features for both target and predictors
   - Optional: stationarity tests and seasonality decomposition
2. Modelling Stage - Predictive Signal Construction
   - 2.1 Model Selection
   - Linear models: OLS, Ridge, Lasso
   - Tree-Based models: XGBoost, LightGBM
   - Sequence models(later): RNN - LSTM, GRU
   - Optional: Prophet or Kalman filters for trend detection
   - 2.2 Model Evaluation (Predictive Quality)
   - RMSE (target < 0.1 for high accuracy; 0.2–0.4 acceptable for macro)
   - R² score (0.7–0.9 is strong; 1.0 may indicate overfitting)
   - Directional accuracy
   - Visual inspection: predicted vs actual
   - 2.3 Hyperparameter Tuning
   - Use GridSearchCV or Optuna with TimeSeriesSplit
   - Optimize RMSE or directional metrics while avoiding overfit
     
3. Strategy Testing Stage - Signal Performance
   - 3.1 Backtesting
   - Use model output to create signals (e.g., yield levels, changes, or directional labels)
   - Evaluate with: Annualised return, Sharpe Ratio, Max Drawdown, Volatility adjusted return (less relevant in bond investing)
   - For signal generation direction: You can use predicted changes in yields (e.g., ΔYield) → generate long/short signals. (中国_中债国债到期收益率_10年)
   - 3.2 Feature Combination Testing
   - Loop through combinations of variables
   - Evaluate each set based on the backtested Sharpe ratio or return
   - For performance evaluation: Apply those signals to the 中债国债总净价指数 to simulate actual investment performance. 
   - Select combinations that perform well in out-of-sample
   - 3.3 Overfitting Detection
   - Use rolling or walk-forward backtesting
   - Compare in-sample vs out-of-sample performance
   - Run robustness checks:Randomized signals, Add cost/slippage, Stress-test signal thresholds
  

# Model Evaluation Metrics
 1. Forecasting Yield / Interest Rate (e.g. 10Y CGB Yield)

- Mean Squared Error (MSE) / Root Mean Squared Error (RMSE): Measures the average prediction error. Sensitive to large deviations.

- Mean Absolute Error (MAE): Less sensitive to outliers than MSE; interpretable in yield terms (bps).

- R² (Coefficient of Determination): Useful to see how much variance your model explains.

✅ Use these if your model is built to predict exact yield levels.

2. Forecasting Signals / Timing Decisions (e.g. +1, 0, -1)
- If you're converting macro or valuation signals into timing indicators:

- Accuracy / Precision / Recall: Using +1 / 0 / -1 frameworks to classify signals.

- Confusion Matrix: To understand the distribution of timing signals (e.g. false positives).

✅ Use if your model outputs directionality or discrete timing signals.

3. Portfolio / Backtesting Evaluation (based on model decisions)
- Since our model leads to actual positioning (e.g. long or short duration), we want trading performance metrics:
- Sharpe Ratio: Measures risk-adjusted return. 
- Maximum Drawdown: critical to assess downside risk in a bond portfolio context.
- Annualised Return / Volatility: to track performance consistency and risk exposure.
- Win rate: percentage of periods the model correctly positioned the trade.

# Datafiles
1. 中國_中債國債到期收益率_10年 - Can be used as the target variable in the predictive model, and also for calculating yield changes, yield spreads, duration, etc.
2. 中債國債總淨價指數 - Serves as a benchmark for backtesting and for verifying whether a strategy generates alpha.
3. 回測結果（兩個） - Used to evaluate the effectiveness of the strategy under different market conditions, and to calculate performance metrics such as Sharpe ratio and maximum drawdown.
4. 因子signal匯總 - Used to generate portfolio allocation signals; signals can also be combined into a composite indicator using factor weighting.
5. 因子匯總 - Used for model training, such as PCA dimensionality reduction, factor selection, or scoring model development.
6. 估值因子signal - Helps determine whether interest rate bonds are “undervalued or overvalued”; considered a valuation-type signal.
7. 宏觀因子signal / 宏觀因子篩選 - Provides macro trend signals (e.g., economic expansion is negative for bonds, signal = -1).
8. 宏觀择時因子庫 - Can be used to generate timing signals based on z-scores, percentiles, or other emission mechanisms.


# Procedures
1. Original data have 20,206 na and 2 columns of  constant variance which has been removed
2. There is 2036 na remained amd 143 numerical columns that are not imputed by pmm for unknown reason (Maybe due to incomplete process). (R)
4. Tried running the pmm with 20 interations, but still ended up with 2036 na remained with unknown reason. (R)
5. The remaining na are replaced by 0. (R) 
6. PCA ran successfully (R)
7. Visualised the data in PCA (R)
8. Export data into rds file (R)
9. Ran 3 linear regressions of the top 10 PCs (Python)
10. 

# ML_python result
- "中国:产量:发电量:当月值.1" 是最核心因子 (It consistently shows strong positive predictive power across all models (coefficient close to 1))
- Ridge is the most robust choice (High R2, low error, and retains multiple contributing features)
- Lasso is useful for feature selection (Effective for initial variable screening or dimensionality reduction, but underperforms slightly in this dataset)
- Linear regression is too perfect and may be overfitting (Suitable as a benchmark, but not ideal for direct deployment)

# Backtesting for Ridge
The Ridge model performs very stably in backtesting. RMSE consistently decreases, and R² approaches perfection, suggesting strong out-of-sample predictive power. Currently, there are no signs of overfitting.

But you should ask yourself:

Is the data too stable, making the model appear ideal?

Will the model retain this level of precision if the data regime changes?

You may want to consider adding noise or perturbations to the data and performing a stress test to assess robustness.

# Note
1. R programming outputed garbled code when execute in csv format. Changed to rds and read successfully in Python
2. Do not move the export line in R. It is because date is added back and move the line forward may lead to error in PCA analysis
