#We deployed long-short term memory (LSTM) in recurrent neural network for our sequantial data
#Optimised the model with the best activation function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyreadr
import statistics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

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
y = pd.DataFrame(yield_data)

#Normalisation
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y)

#Shape the data into 3D samples, timesteps, features

timesteps = 3  # you can adjust this
x_seq, y_seq = [], []
for i in range(timesteps, len(x_scaled)):
    x_seq.append(x_scaled[i - timesteps:i])
    y_seq.append(y_scaled[i])   
x_seq, y_seq = np.array(x_seq), np.array(y_seq)

x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size = 0.2, random_state = 42)

#Input the name of different activation functions
functions = ['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'swish', 'gelu', 'hard_sigmoid', 'softsign', 'softmax', 'linear']

#Run long-short term memory in all activation functions above
for af in functions:
    seq_length=10
    lstm_model = Sequential(
    [LSTM(50, activation = af, input_shape = (x_train.shape[1], x_train.shape[2])),
     Dense(1)
    ])
    lstm_model.compile(optimizer= 'adam', loss='mse')
    train = lstm_model.fit(
    x_train, y_train,
    epochs=50,
    validation_data = (x_test, y_test),
    verbose =0)
    pred = lstm_model.predict(x_test)
    inv_pred = scaler.inverse_transform(pred)
    y_test_rescaled = scaler.inverse_transform(y_test)
    mse = mean_squared_error(y_test_rescaled, inv_pred)
    r2 = r2_score(y_test_rescaled, inv_pred)
    
    print( "The Performance of activation function", af, "under Long-Short Term Memory:" )
    print("RMSE:", np.sqrt(mse))
    print("R2:", r2)  
    print("------------------------------------------")

#Find the best hyperparameters in the best performance activation function
def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Hyperparameter tuning
param_grid = {
    'units': [32, 50, 64],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [8, 16],
    'epochs': [50, 100]
}

best_rmse = float('inf')
best_params = {}
best_model = None

for units in param_grid['units']:
    for dropout in param_grid['dropout']:
        for batch_size in param_grid['batch_size']:
            for epochs in param_grid['epochs']:
                model = build_lstm_model((X_train.shape[1], X_train.shape[2]), units, dropout)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'units': units,
                        'dropout': dropout,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'r2_score': r2
                    }
                    best_model = model

print("Best Parameters:")
for k, v in best_params.items():
    print(f"{k}: {v}")
print("Best RMSE:", best_rmse)
print("Best R²:", best_params['r2_score'])
