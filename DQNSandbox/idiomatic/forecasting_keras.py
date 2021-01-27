# Stock Price Prediction Using Python & Machine Learning
# youtube: https://www.youtube.com/watch?v=QIUxPv5PJOY

# forecasting with LSTM
# import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import keras as keras

import quandl
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True, threshold=1)
# does not do the trick
# pd.set_option('display.max_rows', 1000)
plt.style.use('fivethirtyeight')
def log(value):
    with pd.option_context('display.max_rows', 5000, 'display.max_columns', 50):
        print(value)

print(keras.__version__)

# a variable for predicting 'n' days into the future
forecast_out = 30

df_webReader = web.DataReader('AAPL', data_source='yahoo', start='2000-01-01', end='2019-12-23' )
df_quandl = quandl.get("WIKI/FB")
#print(df)
#print(df.describe())

# log(df)
# log(len(df_quandl))
# log(df_quandl)

'''

'''
open_series = df_quandl.get('Open')
high_series = df_quandl.get('High')
low_series = df_quandl.get('Low')
close_series = df_quandl.get('Adj. Close')

# log(close_series)

'''
another way to get the data
'''
#df_adj_close = df_quandl[['Adj. Close']]
df_adj_close = df_webReader[['Adj Close']]
# log(adj_close)

# create target(column) for dependent variable - shifted n-units up
df_adj_close['Prediction'] = df_webReader[['Adj Close']].shift(-1)
#log(df_adj_close.head(10))

# convert to numpy array
# and create the Adjusted Close based X independent Variable
X = np.array(df_adj_close.drop(['Adj Close'], axis=1))
# log(X)
X = X[:-forecast_out]
#log(X)

# create the dependent variable Y
# also 'wrapped' by numpy array
y = np.array(df_adj_close.drop(['Prediction'], axis=1))
y = y[:-forecast_out]
#log(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("checking the length of datasets")
print("Length of x_train: %s" % format(len(x_train)))
print("Length of x_test: %s" % format(len(x_test)))
print("Length of y_train: %s" % format(len(y_train)))
print("Length of y_test: %s" % format(len(y_test)))

# create and train the SVM(Regressor)
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(x_train, y_train)

# Testing the model: Score returns the coefficient of determination R^2 of the prediction
# best possible score is 1.0
svm_confidence = svr.score(x_test, y_test)
log("confidence is %s" % format(svm_confidence))

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("linear regression has confidence of %s" % format(lr_confidence))

# set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df_adj_close.drop(['Prediction'], 1))[-forecast_out:]
#log("x_forecast values:  %s" % format(x_forecast))
#print(len(x_forecast))

# print linear regression model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
log("LinearRegression predicted values for the next 30 days based on adjusted close: %s" % format(lr_prediction))

svr_prediction = svr.predict(x_forecast)
log("SVM predicted values for the next 30 days based on adjusted close: %s" % format(svr_prediction))



