import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import hashlib
import numpy as np
import sys
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit


# we could also use treshold=np.inf which makes the size of the array at which it is truncated to infinity.
# see discussion: https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array-without-truncation
np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=False)

# from web
download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
s_housing_path = 'datasets/housing'
s_housing_url = download_root+s_housing_path+'/housing.tgz'

def download_then_extract(url=s_housing_url, path=s_housing_path):
    if not os.path.isdir(path):
        os.makedirs(path)
    tgz_file = os.path.join(s_housing_path, 'housing.tgz')
    urllib.request.urlretrieve(s_housing_url, tgz_file)
    housing_tgz = tarfile.open(tgz_file)
    housing_tgz.extractall(path=s_housing_path)
    housing_tgz.close()

'''
We received data in the following format:
longitude | latitude | housing_median_age | total_rooms | total_bedrooms |  population | households | median_income | median_house_value | ocean_proximity
plot the data
'''
def open_file(path=s_housing_path):
    csv_path = os.path.join(path, 'housing.csv')
    data_frame = pd.read_csv(csv_path)
    print(data_frame.head(10))
    return data_frame


download_then_extract(s_housing_url, s_housing_path)
data = open_file(s_housing_path)

# TODO uncomment to show histogramm
# data.hist(bins=50, figsize=(20, 15))
# plt.show()

'''
Note: we are going to access a lot of attributes many times.
even though the file is not 'well structured' or programmed well
it is preferable to access columns and attributes by string constants 
'''
OCEAN_PROXIMITY = "ocean_proximity"


'''
looking at the created histogramm there is a number of observations we can make: 
1. median income does not look like regular US$ eval - instead the data looks skewed or manicured. Indeed looking
up the source we can find out that all below 15K and  
'''

'''
There is no id column for the dataset. reset_index() adds a column 'index' to use as
column index from now on. 
'''
# Add a column index to the dataset
housing_data_with_id = data.reset_index()  # add column index

# self split function
# test_ratio is a value between 0 and 1. default could 0.5 to split a set 50-50
# but it is not a good solution ! over time the function will 'see' the whole dataset which
# we need to avoid.
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    print("shuffled_indices: ", shuffled_indices)

    mean_housing = housing_data_with_id.mean()
    print("mean_", mean_housing)

    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    print("test_set_size: %d" % test_set_size)

    return data.iloc[train_indices], data.iloc[test_indices]


train, test = split_train_test(housing_data_with_id, .2)
print("Length of split train dataset %d" % len(train))
print("Length of split test dataset %d" % len(test))

'''
using a column id as unique identifier we need to secure that all data is appended to the dataset.
And that no column is deleted. 
A different solution is to combine the most stable attributes to a unique id, like long and lat value
or meter_number, area_code and creation timestamp.
 
'''
def split_train_test_validate_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def test_set_check(ident, test_ratio, hash):
    return hash(np.int64(ident)).digest()[-1] < 256 * test_ratio

train_set, test_set = split_train_test_validate_by_id(housing_data_with_id, 0.2, 'index')

print(train_set.head())
print(test_set.head())

'''
an example of a number of attributes to form a stable id
'''
housing_data_with_id['id'] = data["longitude"] * 1000 + data["latitude"]
train_set, test_set = split_train_test_validate_by_id(housing_data_with_id, 0.2, "id")
print("train_set length: %d" % len(train_set))
print("test_set length: %d" % len(test_set))

'''
sci-kit offers a method that already creates a stratified sample.
see strata to find out more on stratified samples
'''
sk_train_set, sk_test_set = train_test_split(data, test_size=0.2, random_state=42) # you know, like the meaning of life, liberty and everything :P
print("sk_train_set length: %d" % len(train_set))
print("sk_test_set length: %d" % len(test_set))

# TODO find and visualize any differences if any on the stratified and self-stratified dataset. experiment!

'''
Since sometime we deal with an ascending numeric value, like median income, but we need a categorical 
representation to be able to stratify the data correctly. 
see: praxis ml with sk learn and tensorflow on kindle. page: 51
'''
housing = data.copy()
print(housing.head())

'''
create a categorical attribute
divide the median_income by 1.5 to limit the possible amount of categories.
and round up with ceil to result in diskrete categories.
'''
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

print(len(housing["income_cat"]))

strat_train_set = []
strat_test_set = []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

print(housing["income_cat"].value_counts())
print(housing["income_cat"].value_counts()/len(housing))

print(" ****** income_cat ****** ")
print(housing["income_cat"].head())

'''
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
'''

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=False)

print(strat_train_set.head())
print(strat_test_set.head())

# chapter 2.10 - visualize the data

# TODO check whether copy creates a shallow or deep copy
housing_vis = strat_train_set.copy()

# TODO uncomment to make visible
# using alpha and a scatter plot it is possible to visualize where median income is located most densely
# TODO experiment with visualisation
# housing_vis.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
''' 
housing_vis.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing_vis["population"]/100, label="population", figsize=(10, 7),
                 c="median_income", cmap=plt.get_cmap("jet"), colorbar=True
                 )

housing_vis.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing_vis["population"]/100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
                 )
plt.legend()
plt.show()
'''

# looking for correlations
# since we are not dealing with big data we can calc the
# correlation coefficient aka pearsons correlation coefficient
corr_matrix = housing.corr()
print("corr matrices:")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print(corr_matrix["population"].sort_values(ascending=False))
print(corr_matrix["median_income"].sort_values(ascending=False))
print(corr_matrix["total_rooms"].sort_values(ascending=False))

# after simply printing
'''
Please note that the main diagonal of the plot would simply plot a number of straight lines if panda were 
to plot each attribute against itself. Instead pandas shows a histogramm.
'''
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

'''
Since what we want to predict is the median house_value and the most promising chart is the median_income we zoom in 
on that.
Note: Looking closely we see that there are horizontal lines of datapoints. These "outliers" need to be removed
else the algo. might 'learn' from them and predict garbage.
'''
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

'''

'''
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix_adjusted = housing.corr()
print("correlation matrix median_house_value")
print(corr_matrix_adjusted["median_house_value"])

'''
Cleaning up the data: 
we need a clean starting set of data. so lets copy strat_train_set again...
'''
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

print("after drop 'median_house_value' axis 1 .....")
print(housing.head(6))

print("printing housing_labels:")
print(housing_labels.head(6))

'''
There are three methods that can be used to clean up the dataFrame.
three options are:
dropna() - drop any NaN value 
drop() - drop the whole column? 
fillna() - fill all NaN values with a median
'''
housing.dropna(subset=["total_bedrooms"])
housing_drop_total_bedrooms = housing.drop("total_bedrooms", axis=1, inplace=False)

'''
this third option needs a median value - housing_median - to be inserted where a NaN has been detected.
'''
housing_median = housing["total_bedrooms"].median()
print(housing_median)
housing_fillna = housing["total_bedrooms"].fillna(housing_median, inplace=False)

print("housing_fillna: ")
print(housing_fillna)
print(type(housing_fillna))

# housing_fillna.plot(kind="scatter", x="total_rooms", y="total_bedrooms", alpha=0.1)
# plt.show()

# Time to move on ...
'''
Any time we haved to deal with missing values we can use a small utility
called an "Imputer". The imputer allows to replace a set of missing numbers.

'''
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")  # also a strategy could be median
housing_num_only = housing.drop(OCEAN_PROXIMITY, axis=1)
imputer.fit(housing_num_only)

print(imputer.statistics_)
print(housing_num_only.median().values)

# now that the imputer has been 'trained' we can use it to transform the new dataset, so missing values
# can be replaced by found median values.
X = imputer.transform(housing_num_only)
# now store the result in a DataFrame
housing_trained_df = pd.DataFrame(X, columns=housing_num_only.columns)

print("columns:")
print(housing_trained_df.columns)

print("head:")
print(housing_trained_df.head())
print("tail:")
print(housing_trained_df.tail())

'''
Modify and process text and categorical attributes
ocean_proximity was left out earlier since we cannot calculate a median value 
from a categorical attribute.
'''
housing_cat = housing[OCEAN_PROXIMITY]
print(housing_cat.head(10))


# since most algorithms we will use like numbers anyway,
# let's see whether we can convert the category to numbers
housing_cat_encoded, housing_categories = housing_cat.factorize()

print("result last ten of factorize:")
print(housing_cat_encoded[:10]) # show last ten elems

'''
A challenge when dealing with categorical attributes is that a factorize or a quantification does not 
reflect its correlations to each other. 
Given our categorical values as housing_categories:
'''
print("categorical attributes: ")
print(housing_categories)

'''
an algorithm will have trouble quantifying the different values.
A value at 1 is closer to 4, than 2 to 4. (given the list starts at 0 ) 
'<1H OCEAN', 'NEAR OCEAN', 'INLAND', 'NEAR BAY', 'ISLAND'
To get around a OneHot-Encoding is used where a binary attribute is used per category.

'''

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot_encoded = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1)) # remember housing_cat_encoded from line 308 :)
# now lets look at the result:
print("housing_cat_1hot_encoded: ")
print(housing_cat_1hot_encoded.toarray())
print("typeof housing_cat_1hot_encoded: ")
print(type(housing_cat_1hot_encoded))


















