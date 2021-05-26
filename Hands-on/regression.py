import os
import tarfile
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import scipy.stats as stats

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()

# print(housing.head())
# print(housing.info())
# print(housing["total_rooms"].value_counts())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# def split_train_test(data, test_ratio):  # not good for updating database
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(test_ratio * len(data))
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
#
# def test_set_check(identifier, test_ratio):
#     """ needs special columns for id """
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * (2 ** 32)
#
#
# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
#     return data.iloc[~in_test_set], data.iloc[in_test_set]
#

# housing_with_id = housing.reset_index()

""" this is from sklearn"""
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set))
# print(len(test_set))

""" splitting based on median income and 5 stratus."""
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# housing["income_cat"].hist(bins=70, figsize=(20, 15))
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

""" checking if it is really split accordingly to median income """
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

""" dropping income_cat no need from now """
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

""" showing data in a nice way """
housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude")
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 50, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
# plt.legend()
# plt.show()

""" computing correlation """
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

""" getting read of empty cells by deleting whole rows"""
# print(housing.info())
# housing = housing.dropna(subset=["total_bedrooms"])
# housing.drop("total_bedrooms", axis=1)
# print(housing.info())

""" filling empty cells with data """
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])

""" one hot encoding """
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
""" all data prepared! """

""" modeling part (Linear Regression)"""
# print(housing_prepared)
# print(housing_prepared.head())
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse) not good ;-;

""" modeling with Tree Regressor """
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# display_scores(tree_rmse_scores)

# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scroes = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scroes)

""" modeling with Forest Regressor """

# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# print(forest_mse)
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

""" final model """
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

""" checking Support Vector Machine Regressor """
svr = SVR(kernel='rbf')
# param_grid = {'C': [0.1, 1, 100, 1000], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'degree': [1, 2, 3, 4, 5, 6],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
param_grid_for_rand = {'C': stats.uniform(0, 100), 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                       'degree': [1, 2, 3, 4, 5, 6],
                       'gamma': stats.uniform(0, 100)}
# grid_search = GridSearchCV(svr, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search = RandomizedSearchCV(svr, param_grid_for_rand, n_iter=10)
grid_search.fit(housing_prepared, housing_labels)
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
