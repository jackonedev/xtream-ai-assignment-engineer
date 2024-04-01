# Last_version: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/pipeline.py
# Version_used: https://github.com/scikit-learn/scikit-learn/blob/1.2.X/sklearn/pipeline.py

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class PandasFeatureUnion(FeatureUnion):
    """
    PandasFeatureUnion inherit from sklearn FeatureUnion to create a pipeline \
for feature engineering that return pandas.DataFrame.
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans, X=X, y=y, weight=weight, **fit_params
            )
            for name, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    # def transform(self, X, y=None, **params):
    #     self._validate_transformers()
    #     result = Parallel(n_jobs=self.n_jobs)(
    #         delayed(_transform_one)(
    #             transformer=trans, X=X, y=y, weight=weight, params=params
    #         )
    #         for name, trans, weight in self._iter()
    #     )

    #     if not result:
    #         # All transformers are None
    #         return np.zeros((X.shape[0], 0))
    #     Xs, transformers = zip(*result)
    #     self._update_transformer_list(transformers)
    #     if any(sparse.issparse(f) for f in Xs):
    #         Xs = sparse.hstack(Xs).tocsr()
    #     else:
    #         Xs = self.merge_dataframes_by_column(Xs)
    #     return Xs

    # def transform(self, X):
        # Xs = Parallel(n_jobs=self.n_jobs)(
        #     delayed(_transform_one)(transformer=trans, X=X, y=None, weight=weight)
        #     for name, trans, weight in self._iter()
        # )
        # if not Xs:
        #     # All transformers are None
        #     return np.zeros((X.shape[0], 0))
        # if any(sparse.issparse(f) for f in Xs):
        #     Xs = sparse.hstack(Xs).tocsr()
        # else:
        #     Xs = self.merge_dataframes_by_column(Xs)
        # return Xs

    def transform(self, X, **transformer_params):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans, X=X, y=None, weight=weight, params=transformer_params
            )
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
    ## INTENTO 3
    # def transform(self, X):
    #     # Obtener transformadores junto con sus respectivos pesos y parámetros
    #     transformers = self._iter()
    #     print(list(transformers))
    #     Xs = Parallel(n_jobs=self.n_jobs)(
    #         delayed(_transform_one)(
    #             transformer=trans, X=X, y=None, params=transformer_params
    #         )
    #         for name, trans, transformer_params in transformers
    #     )
    #     if not Xs:
    #         # All transformers are None
    #         return np.zeros((X.shape[0], 0))
    #     if any(sparse.issparse(f) for f in Xs):
    #         Xs = sparse.hstack(Xs).tocsr()
    #     else:
    #         Xs = self.merge_dataframes_by_column(Xs)
    #     return Xs