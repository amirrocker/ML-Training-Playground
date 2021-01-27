import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib


DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
DOWNLOAD_QUANDL_METADATA = "https://www.quandl.com/api/v3/datasets/WIKI/FB.json?api_key=xzWxocPDLm3Qjr3RNcyW"

HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

METADATA_PATH = "datasets/quandl/"
METADATA_URL = DOWNLOAD_QUANDL_METADATA


def fetch_data(url, path, filename='housing.tgz'):
    if not os.path.isdir(path):
        os.makedirs(path)
    tgz_path = os.path.join(path, filename)
    urllib.request.urlretrieve(url, tgz_path)

    if tarfile.is_tarfile(tgz_path) is False:
        data = pd.read_csv(tgz_path)
        print(data)
    else:
        data_gzp = tarfile.open(tgz_path)
        data_gzp.extractall(path=path)
        data_gzp.close()
        print(data_gzp)


def load_data(path=HOUSING_PATH, filename='housing.csv'):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)

def plot_hist(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

# fetch_data(HOUSING_URL, HOUSING_PATH)
fetch_data(METADATA_URL, METADATA_PATH, 'metadata.csv')
csv = load_data(METADATA_PATH, "metadata.csv")
# csv = load_data(HOUSING_PATH)
print(csv.info())
plot_hist(csv)

#if __name__ == "__main__":

# factor modeling
idx = pd.IndexSlice
with pd.HDFStore('/datasets/assets.h5') as store:
    prices = store['quandl/wiki/prices'].loc[idx['2000':'2018', :],
                    'adj_close'].unstack('ticker')
print(prices.info())




