'''
ARIMA Forecast on datasets
source: ARIMA and Python: Stock Price Forecasting using statsmodels
url: https://www.youtube.com/watch?v=o7Ux5jKEbcw&list=PLyQ9mKBq5l7IGGXi582aRDDxrM5a-XCAq&index=54&t=10s
'''

import os
import pandas
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

path = "C:/Users/info/Documents/dataset/"
os.chdir(path)
os.getcwd()
print(path)

variables = pandas.read_csv('Dow_Jones_DJI_1985_2019.csv')
open = variables['Open']
high = variables['High']
low = variables['Low']
close = variables['Close']

print("Open: ", open)
print("high: ", high)
print("Low: ", low)
print("Close: ", close)

# get the logarithm of the price to better discern price movement
logOpenPrices = np.log(open)
plt.plot(logOpenPrices)
plt.show()

acf_1 = acf(logOpenPrices)[1:20]
#plt.plot(acf_1)
#plt.show()

test_df = pandas.DataFrame([acf_1]).T
test_df.columns = ['Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')
plt.show()


pacf_1 = pacf(logOpenPrices)[1:20]
test_df = pandas.DataFrame([pacf_1]).T
test_df.columns = ["Partial Autocorrelation"]
test_df.index += 1
test_df.plot(kind="bar")
plt.show()
