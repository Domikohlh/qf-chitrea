#Voting and XGBoost are selected for the best model

import pandas as pd
import numpy as np
import pyreadr
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# Load macroeconomic data
data = pyreadr.read_r('/Users/dominicevergarden/Desktop/qf_chitrea/aligned_marco_data.rds')
df = data[None]

# Select specified features (remove duplicates)
selected_features = [
    "中国:社会消费品零售总额:当月值", "中国:M1:单位活期存款", "中国:M2", "中国:产量:发电量:当月值.1",
    "中国:M1", "中国:金融机构:企业存款余额", "中国:金融机构:短期贷款余额",
    "中国:城镇居民平均每百户拥有量:家用汽车", "中国:M0", "中国:产量:发电量:当月值"
]
x = df[selected_features]

# Load monthly yield data
yield_df = pd.read_excel('/Users/dominicevergarden/Desktop/qf_chitrea/monthly_avg_yield.xlsx')
y = yield_df['monthly_avg_yield']

# Reset both indices and assign a shared monthly index
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

# Create a shared monthly index (starting from the known start date of your data)
start_date = '2002-01-01'  # adjust if needed
date_index = pd.date_range(start=start_date, periods=len(x), freq='M')
x.index = date_index
y.index = date_index

# Rename feature columns
x.columns = [f"feature_{i}" for i in range(x.shape[1])]

# Split dataset (no shuffle because it's time series)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# Train XGBoost model
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    tree_method='hist',
    verbosity=0
)
model.fit(x_train, y_train)

# Generate predictions
y_pred = model.predict(x_test)

# Generate long/short signals based on prediction change direction (1 = long, -1 = short)
pred_diff = np.diff(y_pred, prepend=y_pred[0])
signal = np.where(pred_diff < 0, 1, -1)

# Calculate benchmark monthly return from actual yield
benchmark_return = y.pct_change().dropna()
print(len(benchmark_return))

# Align benchmark with signal
benchmark_return = benchmark_return.iloc[-len(signal):]
print(len(benchmark_return))

# Calculate strategy return
strategy_return = benchmark_return.values * signal[:len(benchmark_return)]

# Evaluation metrics
ann_return = np.mean(strategy_return) * 12
#Volatility is the risk loss 
volatility = np.std(strategy_return) * np.sqrt(12)
#Shape ratio is the return of receiving one unit of risk (Unlimit drawdown)
sharpe_ratio = ann_return / volatility
cumulative_return = np.cumsum(strategy_return)
running_max = np.maximum.accumulate(cumulative_return)
drawdown = running_max - cumulative_return
#Max
max_drawdown = np.max(drawdown)

# Output results
print("Strategy Backtest Performance(XGBoost):")
print(f"Annualised Return: {ann_return:.4f}")
print(f"Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")

#For Voting Regressor
# Define base models
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
lgbm = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Voting Regressor
model = VotingRegressor(estimators=[
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('rf', rf),
    ('gb', gb)
])

# Train the ensemble model
model.fit(x_train, y_train.values.ravel())

# Generate predictions
y_pred = model.predict(x_test)

# Generate long/short signals based on prediction change direction
pred_diff = np.diff(y_pred, prepend=y_pred[0])
signal = np.where(pred_diff < 0, 1, -1)

# Calculate benchmark monthly return from actual yield
benchmark_return = y.pct_change().dropna()

# Align benchmark with signal
benchmark_return = benchmark_return.iloc[-len(signal):]

# Calculate strategy return
strategy_return = benchmark_return.values * signal[:len(benchmark_return)]

# Evaluation metrics
ann_return = np.mean(strategy_return) * 12
volatility = np.std(strategy_return) * np.sqrt(12)
sharpe_ratio = ann_return / volatility
cumulative_return = np.cumsum(strategy_return)
running_max = np.maximum.accumulate(cumulative_return)
drawdown = running_max - cumulative_return
max_drawdown = np.max(drawdown)

# Output results
print("Strategy Backtest Performance (Voting Regressor):")
print(f"Annualised Return: {ann_return:.4f}")
print(f"Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
