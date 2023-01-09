# Was tempted to call this ItchScratcher ;P

'''
The ITCH parser depends on a message specification provided by a .csv file
created by script create_message_specification.py

'''

import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin

# from clint.textui import progress


data_path = Path('data')  # choose a good path!
itch_store = str(data_path / 'itch.h5')  # stored as hdf format (HDF5)
order_book_store = data_path / 'order_book.h5'

# now download the data
FTP_URL = "ftp://emi.nasdaq.com/ITCH/Nasdaq ITCH/"
HTTPS_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
# SOURCE_FILE = '03272021.NASDAQ_ITCH50.gz'
SOURCE_FILE = '12282018.NASDAQ_ITCH50.gz'


def try_download(url):
    print("try download started ... ")
    '''download and unzip data if not yet available'''
    filename = data_path / url.split("/")[-1]
    print(filename)
    if not data_path.exists():
        print("create directory")
        data_path.mkdir()
    if not filename.exists():
        print("downloading....", url)
        urlretrieve(url, filename, reporthook=download_progress_hook)


def download_progress_hook(count, blockSize, totalSize):
    print(f'downloaded: {count/1024} MB , loading: {blockSize/1024} KB, totalSize: {totalSize/1024/1024} MB')


'''

formats = {
    ('integer', 2): 'H',
    ('integer', 4): 'I',
    ('integer', 6): '6s',
    ('integer', 8): 'Q',

    ('alpha', 1): 's',
    ('alpha', 2): '2s',
    ('alpha', 4): '4s',
    ('alpha', 8): '8s',
    ('price_4', 4): 'I',
    ('price_8', 8): 'Q',
}

# get ITCH specs
specs = pd.read_csv('message_types.csv')

specs['formats'] = specs[['value', 'length']].apply(tuple, axis=1).map(formats)

# formatting alpha fields
alpha_fields = specs[specs.value == 'alpha'].set_index('name')
alpha_msgs = alpha_fields.groupby('message_type')
alpha_formats = {k: v.to_dict() for k, v in alpha_msgs.formats}
alpha_length = {k: v.to_dict for k, v in alpha_msgs.length}

# generate message classes as named tuples and format strings
message_fields, fstring = {}, {}
for typename, message in specs.groupby('message_type'):
    message_fields[t] = namedtuple(
        typename=typename,
        field_names=message.name.tolist())
    fstring[typename] = '>' + ''.join(message.formats.tolist())

print(message_fields)

def format_alpha(mtype, data):
    for col in alpha_formats.get(mtype).keys():
        # stock name only in summary message 'R'
        if mtype != 'R' and col == 'stock':
            data = data.drop(col, axis=1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(encoding.get(col))
    return data
'''

if __name__ == "__main__":
    print("main called")
file_name = try_download(urljoin(HTTPS_URL, SOURCE_FILE))
date = file_name.name.split('.')[0]
print(date)
print(file_name)
