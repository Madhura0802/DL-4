# DL-4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


dataset = pd.read_csv('GOOGL.csv', index_col='Date', parse_dates=['Date'])


train_set = dataset[:'2018'].iloc[:, 1:2].values
test_set = dataset['2019':].iloc[:, 1:2].values


sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)


def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        Y.append(dataset[i, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y


look_back = 60
X_train, Y_train = create_dataset(train_set_scaled, look_back)
X_test, Y_test = create_dataset(test_set, look_back)


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, Y_train, epochs=100, batch_size=32)


mse = model.evaluate(X_test, Y_test)
rmse = np.sqrt(mse)



last_60_days = train_set[-60:]
last_60_days_scaled = sc.transform(last_60_days)
X_pred = np.array([last_60_days_scaled])
X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
pred_price = model.predict(X_pred)
pred_price = sc.inverse_transform(pred_price)
print(pred_price)

