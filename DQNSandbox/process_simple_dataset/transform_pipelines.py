'''
Note the encoder fit_transform function requires a 2D-Array. But since housing_cat_encoded is only a 1D array the
need to reshape the array. And result is a sparse matrix no numpy array.
the array form shows the use of OneHotEncoding quite well.
to get the above processing in one step sklearn offers a CategoricalEncoder class. It is at this point not yet
part of sklearn but available on Github - PR #9151
'''
# TODO include the PR - or use a ColumnTransformer as suggested by #9012
# https://github.com/scikit-learn/scikit-learn/pull/9012
# TODO and also
# https://github.com/scikit-learn/scikit-learn/pull/8793

'''
We came across Pipeline code in sklearn so its time to look at some transformers of our own.
sklearn offers a diverse set of transformers but sometimes we do need our own:
scikit learn using Duck Typing all that needs to be done is to define a class and implement
three methods to conform to sklearn api.
If we extend TransformerMixin the last of the three methods is handled automatically.
We could use BaseEstimator as another Super class and get more methods like get_params and set_params.  
'''
# TODO look up DuckTyping vs. inheritance

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bed_rooms_per_room=True):  # make sure to not pass *args or **kargs - TODO look up **kargs
        self.add_bedroom_per_room = add_bed_rooms_per_room

    def fit(self, X, y=None):
        return self  # since we do not train anything simply return

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]
        # this is where the meat is on the bone ;)
        population_per_household = X[:, population_ix] / X[:, household_ix]
        print("calculated population_per_household: ")
        # print(population_per_household)

        if self.add_bedroom_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            '''
            From the documentation of CClass - AxisConcatenator in numpy/libs/index_tricks.py

               Translates slice objects to concatenation along the second axis.

               This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
               useful because of its common occurrence. In particular, arrays will be
               stacked along their last axis after being upgraded to at least 2-D with
               1's post-pended to the shape (column vectors made out of 1-D arrays).

               See Also
               --------
               column_stack : Stack 1-D arrays as columns into a 2-D array.
               r_ : For more detailed documentation.

               Examples
               --------
               >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
               array([[1, 4],
                      [2, 5],
                      [3, 6]])
               >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
               array([[1, 2, 3, ..., 4, 5, 6]])

            '''
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# now try out the new class:

attribute_adder = CombinedAttributesAdder(add_bed_rooms_per_room=False)
housing_extra_attributes = attribute_adder.transform(housing.values)
print("combined attributes: ")
print(housing_extra_attributes)

'''
Scaling of attributes:
important transforms include scaling. most ml algorithms do not play well with numeric values with differejnt scales.
Look at the data for housing - the numbers of rooms per household quite differ from median_income. Any algorithm 
would have its difficulties learning from this data set. Note that we need no scaling of the target attributes.

there are two default scaling methods - min-max scaling and standardising.
min-max scaling aka normalising = scale all values from 0 to 1 
standardising =  
'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="median")),
        # remember the simple imputer from above? replacing missing values
        ('attributes_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ]
)
num_pipeline.fit_transform(housing_num_only)
print("housing_num_only: ")
print(housing_num_only)
print(housing_num_only.columns)

'''
Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'income_cat'],
'''
# housing_num_only.plot(kind="scatter", x="median_income", y="total_bedrooms")
# housing_num_only.boxplot("median_income")
# housing_num_only.hist("median_income")
# plt.show()

''' 
A short excourse into Matplotlib - uncomment for plot drawing
'''
# fig = plt.figure()
# axis = fig.add_subplot(111) # 111 = rows - cols - num
# axis.scatter(housing_num_only["income_cat"], housing_num_only["median_income"], color="darkgreen", marker="^")
# axis.plot(housing_num_only["income_cat"], housing_num_only["median_income"], color="lightblue", linewidth=3)
# plt.show()

### DataFrames
'''
It would be preferable to be able to pass in a dataframe instead of having to turn the data into a numpy array first.
'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


number_attributes = list(housing_num_only)
category_attributes = [OCEAN_PROXIMITY]

# create two necessary pipelines
# one to prepare numerical the other categorical data
numerical_pipeline = Pipeline([
    ('selector', DataFrameSelector(number_attributes)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attributes_adder', CombinedAttributesAdder()),
    ('standard_scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(category_attributes)),
    ('label_binarizer', LabelBinarizer())
])

feature_pipeline = FeatureUnion(transformer_list=[
    ('numerical', numerical_pipeline),
    ('categorical', categorical_pipeline)
])

housing_data_prepared = feature_pipeline.fit_transform(housing_data_with_id)
print("housing_data_prepared: ")
print(housing_data_prepared)