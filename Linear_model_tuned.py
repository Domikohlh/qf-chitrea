#Import library
#Install all the libraries with 'pip install (Library)'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, mean_squared_error, accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import pyreadr
import statistics

#Read rds data
data = pyreadr.read_r('/cleaned_macro_data.rds')

#Extract pandas dataframe
df = data[None]
#Extract pandas dataframe
df = data[None]
top9_PC = [
    "中国:社会消费品零售总额:当月值", "中国:M2", "中国:产量:发电量:当月值.1",
    "中国:M1", "中国:金融机构:企业存款余额", "中国:金融机构:短期贷款余额",
    "中国:城镇居民平均每百户拥有量:家用汽车", "中国:M0", "中国:产量:发电量:当月值"
]
x = pd.DataFrame(df[top9_PC])
y = pd.DataFrame(df.loc[:, "中国:M1:单位活期存款"])

#Calculate the standard deviation for each columns
for feature in x.columns:
    col_data = x[feature]
    sdv = statistics.stdev(col_data)
    print("The standard deviation of", feature, "is :", "\n", sdv)
    print("----------------------------------")

#Split the training and testing data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#Input the linear (base) model
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
lasso_model = Lasso(alpha=0.1)
lin_model = LinearRegression()

for ml in (ridge_model, lasso_model, lin_model):
    ml.fit(x_train, y_train)
    model_preds = ml.predict(x_test)
    mse = mean_squared_error(y_test, model_preds)
    print(ml.__class__.__name__, "Performance:")
    print("RMSE:", np.sqrt(mse) )
    print("R2 :", r2_score(y_test, model_preds))
    print("Intercept (Theta_0):", ml.intercept_)
    print("Coefficient (Theta_1):", ml.coef_)
    plt.plot(model_preds, model_preds, 'r-', zorder = 2)
    plt.plot(x,y, 'b.', zorder =1)
    plt.title(ml.__class__.__name__)
    plt.xlabel('Retail price')
    plt.ylabel('Predictions')
    plt.show()
    print("----------------------------------")

#Standardisation
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Load regularised pure model
ridge_model = RidgeCV()
lasso_model = Lasso()

#Set parameter grid for 2 models

ridge_params = {
    'alphas': [0.01, 0.1, 1, 10 ,100]
}

ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=5, scoring = 'neg_mean_squared_error', verbose =1)

lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],  # Smaller alphas than Ridge
}

lasso_grid = GridSearchCV(lasso_model, lasso_params, cv=5, scoring = 'neg_mean_squared_error', verbose =1)

#Hyperparameter for 2 regularised model

for grid in (ridge_grid, lasso_grid):
    grid.fit(x_train_scaled, y_train)
    print("Best", grid ,"Params:", grid.best_params_)
    best_estim = grid.best_estimator_
    tuned_preds = best_estim.predict(x_test_scaled)
    
    print("MSE:", mean_squared_error(y_test, tuned_preds))
    print("R2:", r2_score(y_test, tuned_preds))
    plt.plot(y_test.values, tuned_preds, 'bo', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.title("Tuned Model")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()
