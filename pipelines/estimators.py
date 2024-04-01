from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import pickle
import datetime as dt


class RemoveOutliers(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert X[X["price"] > 0].shape[0] > 0, "All values are outliers. Please check the data."
        return X[X["price"] > 0].reset_index(drop=True)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.encoder.fit(self.X_[self.columns])
        return self

    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.encoder.transform(X_encoded[self.columns])
        X_encoded = pd.DataFrame(X_encoded, columns=self.columns)
        return X_encoded


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.columns_ = None
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.encoder.fit(self.X_[self.columns])
        return self

    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.encoder.transform(X_encoded[self.columns])
        X_encoded = self.feature_adjust(X_encoded)
        self.columns_ = X_encoded.columns
        return X_encoded

    def feature_adjust(self, X_encoded):
        result = pd.DataFrame()
        last_len = 0
        for i, features in enumerate(self.encoder.categories_):
            len_feature = len(features)
            formated_features = [
                f"{self.columns[i]}_{feat}".replace(" ", "_") for feat in features
            ]
            builded_features = pd.DataFrame(
                X_encoded[:, last_len : last_len + len_feature],
                columns=formated_features,
            )
            result = pd.concat([result, builded_features], axis=1)
            last_len += len_feature

        result.columns = result.columns.str.replace(
            r"[^\w\s]", "_", regex=True
        ).str.replace("__+", "_", regex=True)
        return result


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=MinMaxScaler):
        self.scaler = scaler()

    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.scaler.fit(self.X_)
        return self

    def transform(self, X):
        X_encoded = X.copy()
        X_encoded = self.scaler.transform(X_encoded)
        X_encoded = pd.DataFrame(X_encoded, columns=self.X_.columns)
        return X_encoded
