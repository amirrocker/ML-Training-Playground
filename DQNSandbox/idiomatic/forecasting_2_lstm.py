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

import matplotlib as plt
import keras as keras

import quandl
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

np.set_printoptions(suppress=True, threshold=1)
# does not do the trick
# pd.set_option('display.max_rows', 1000)
#plt.style.use('fivethirtyeight')


def log(value):
    with pd.option_context('display.max_rows', 5000, 'display.max_columns', 50):
        print(value)


print(keras.__version__)


def get_type(type):
    if isinstance(type, list):
        return 0
    if isinstance(type, np.ndarray):
        return 1
    if isinstance(type, pd.Series):
        return 2
    if isinstance(type, pd.DataFrame):
        return 3


def plotData(data):
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')

    # if data['Adj Close'] is not None:
    if get_type(data) == 0:
        print("list not supported yet...")
    elif get_type(data) == 1:
        plt.plot(data)
    elif get_type(data) == 2:
        plt.plot(data['Adj. Close'])
    else:
        log("Could not retrieve column ... check column name")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel('Adj- Close Price USD($)', fontsize=18)
    plt.show()


# our standard first dataset
df = web.DataReader('AAPL', data_source='yahoo', start='2000-01-01', end='2019-12-22')
# print(df.shape)
# log(df)
# plotData(df)

# create a new dataframe with only the Adj.Close columns
data = df.filter(['Adj Close'])
# convert to numpy array
dataset = data.values
# log(dataset.dtype)

# get the number of rows to train on - calculate as if it were split
training_data_length = math.ceil(len(dataset) + .8)
log(training_data_length)

# create a scaler to remove the Outliers  ?
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
# log(scaled_data)
# plotData(scaled_data)

# x_train, x_test, y_train, y_test = train_test_split(scaled_data)

train_data = scaled_data[0:training_data_length, :]
x_train = []  #
y_train = []  # labels ?

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        log(x_train)
        log(y_train)
        log(len(x_train[0]))
        log(len(y_train))

# log(len(x_train))

# convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
log(x_train.shape)

# Build the LSTM Model using tensorflow - pytorch implementation
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing data set
# create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_length - 60: , :]

# create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_length:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
log("created dataset for x_test: " % format(x_test))

# convert to numpy array
x_test = np.array(x_test)

# reshape the data
x_test = np.reshape(x_test.shape[0], x_test.shape[1], 1)
log(x_test)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_train) ** 2)
log("Root mean squared error: " % format(rmse))