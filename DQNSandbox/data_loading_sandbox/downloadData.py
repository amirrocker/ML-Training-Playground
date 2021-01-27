import os
import tarfile
import pandas as pd
import numpy as np

from six.moves import urllib
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=1000, suppress=False, precision=100000)
pd.options.display.max_columns = None
pd.options.display.max_rows = None


DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing/"
FILE_NAME = "housing.tgz"
CSV_FILE_NAME = "housing.csv"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + FILE_NAME

# Step 1: Download, save and extract Pandas DataFrame

def fetchData(housingUrl=HOUSING_URL, housingPath=HOUSING_PATH):
    if not os.path.isdir(housingPath):
        os.makedirs(housingPath)
    tgzPath = os.path.join(housingPath, FILE_NAME)
    urllib.request.urlretrieve(housingUrl, tgzPath)
    dataTgz = tarfile.open(tgzPath)
    dataTgz.extractall(path=housingPath)
    dataTgz.close()

fetchData()

# now read the created csv file
def loadData(dataPath=HOUSING_PATH):
    csvPath = os.path.join(dataPath, CSV_FILE_NAME)
    return pd.read_csv(csvPath)

data = loadData()
# print("data: %s" % data.head(10))
# print("data: %s" % data.tail(10))

# Step 2: Create a Test data set
def split_train_test(data, test_ratio):
    # shuffledIndices = np.random.permutation(len(data))
    train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=42)
    return train_set, test_set

trainSet, testSet = split_train_test(data, 0.2)
print("trainSet: %s" % trainSet)
print("testSet: %s" % testSet)

data['income_cat'] = np.ceil(data['median_income'] / 1.5)
print("data['income_cat']: %s" % data['income_cat'])



