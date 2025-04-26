#We deployed 3 boosting models (XGBoost, LightGBM and Voting Regressor (XGB + Light + GradB + Random Forest) for yield prediction

import pandas as pd
import numpy as np
import pyreadr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, mean_squared_error, accuracy_score, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

#Read rds data
data_1 = pyreadr.read_r('/Users/dominicevergarden/Desktop/qf_chitrea/aligned_marco_data.rds')
data_2 = pd.read_excel('/Users/dominicevergarden/Desktop/qf_chitrea/monthly_avg_yield.xlsx')

#Extract pandas dataframe
df = data_1[None]

top10_PC = [
    "中国:社会消费品零售总额:当月值", "中国:M1:单位活期存款", "中国:M2", "中国:产量:发电量:当月值.1",
    "中国:M1", "中国:金融机构:企业存款余额", "中国:金融机构:短期贷款余额",
    "中国:城镇居民平均每百户拥有量:家用汽车", "中国:M0", "中国:产量:发电量:当月值"
]

yield_data = data_2.loc[:,'monthly_avg_yield']

x = pd.DataFrame(df[top10_PC])
x.columns = [f'feature_{i}' for i in range(x.shape[1])]
y = pd.DataFrame(yield_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Fit XGBoost model
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
scores = cross_val_score(xgb, x, y, cv=5)

# Output performance
print("XGBoost Performance:")
print("RMSE:", np.sqrt(mse))
print(f"R² Score: {r2:.4f}")
print("Cross Validation Score:", scores)

#Feature Importance
xgb.plot_importance(model)
plt.show

# Fit LightGBM model
lgbm = LGBMRegressor(objective='regression', n_estimators=100, max_depth=3, learning_rate=0.1)
lgbm.fit(x_train, y_train.values.ravel())

# Predict and evaluate
y_pred = lgbm.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output performance
print("LightGBM Performance:")
print("RMSE:", np.sqrt(mse))
print(f"R² Score: {r2:.4f}")

#Voting Regressor

# Define base models
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
lgbm = LGBMRegressor( n_estimators=100, max_depth=3, learning_rate=0.1)

# Voting Regressor
vtm = VotingRegressor(estimators=[
    ('rf', rf),
    ('gb', gb),
    ('xgb', xgb),
    ('lgbm', lgbm)
])

# Fit the ensemble model
vtm.fit(x_train, y_train.values.ravel())

# Predict and evaluate
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
scores = cross_val_score(vtm, x, y, cv=5)

# Output performance
print("Voting Regressor Performance:")
print("RMSE:", np.sqrt(mse))
print(f"R² Score: {r2:.4f}")
print("Cross Validation Score:", scores)
