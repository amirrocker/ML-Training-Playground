import pandas as pd
import pandas_datareader.data as web

pd.set_option('display.expand_frame_repr', 'False')

'''
A number of data loading mechanisms as well as sites where 
data can be retrieved.
Remember that a API KEY may be needed.
quandl: xzWxocPDLm3Qjr3RNcyW
url: "https://www.quandl.com/api/v3/datasets/WIKI/FB.json?api_key=xzWxocPDLm3Qjr3RNcyW"

'''

# define path to the datafile

def get_wiki_prices():
    '''
    source: https://www.quandl.com/api/v3/datatables/WIKI/PRICES?qopts.export=true&api_key=<API_KEY>
    Download and rename to wiki_prices.csv
    :return:
    '''
    df = (pd.read_csv('datasets/wiki_prices.csv',
                     parse_dates=['date'],
                     index_col=['date', 'ticker'],
                     infer_datetime_format=True)
          .sort_index()
          )
    print(df.info(null_counts=True))
    with pd.HDFStore('assets.h5') as store:
        store.put('quandl/wiki/prices', df)
        print(store.info())

    df = pd.read_csv

get_wiki_prices()




