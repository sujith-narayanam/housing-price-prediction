import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(
        self, add_bedrooms_per_room=True,
    ):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = 7
        self.bedrooms_ix = 6
        self.population_ix = 5
        self.households_ix = 0

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


class data_transformer:
    def __init__(self, num_attribs, cat_attribs):
        self.num_columns = num_attribs
        self.cat_columns = cat_attribs
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("attribs_adder", CombinedAttributesAdder()),
                ("std_scaler", StandardScaler()),
            ]
        )

        self.full_pipeline = ColumnTransformer(
            [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)]
        )

    def fit(self, X):
        self.full_pipeline.fit_transform(X)

    def transform(self, X):
        column_names = self.num_columns + [
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ]
        column_names += list(
            self.full_pipeline.transformers_[1][1].get_feature_names_out(
                self.cat_columns
            )
        )
        return pd.DataFrame(self.full_pipeline.transform(X), columns=column_names)
