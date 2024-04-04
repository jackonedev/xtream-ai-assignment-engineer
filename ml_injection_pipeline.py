from typing import Any
import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from pydantic import ValidationError

from schemas.diamond import Diamond
from utils.config import MODELS_PATH, DATASET_PATH


# LOAD THE MODELS AND TRANSFORM PIPELINE

with open(os.path.join(MODELS_PATH, "transform_pipeline.pkl"), "rb") as file:
    transform_pipeline = pickle.load(file)

with open(os.path.join(MODELS_PATH, "log_reg.pkl"), "rb") as file:
    log_reg = pickle.load(file)

with open(os.path.join(MODELS_PATH, "lgbm_common.pkl"), "rb") as file:
    lgbm_common = pickle.load(file)

with open(os.path.join(MODELS_PATH, "lgbm_exclusive.pkl"), "rb") as file:
    lgbm_exclusive = pickle.load(file)


def data_transformation(data: Any) -> pd.DataFrame:

    if isinstance(data, dict):
        try:
            Diamond(**data)
        except ValidationError as e:
            print("Error transforming the input:")
            for element in e.errors():
                element.pop('input')
                element.pop('url')
                print(element)
            raise e

        data = pd.DataFrame(data, index=[0])

    elif isinstance(data, type(np.array(0))) or isinstance(data, list):
        cols = pd.read_csv(DATASET_PATH).drop(columns='price').columns
        data = {col: [val] for col, val in zip(cols, data)}
        data = pd.DataFrame(data)

    data = transform_pipeline.transform(data)
    return data


def prediction_pipeline(data: Any) -> float:
    """
    Function to predict the price of a diamond based on the input data.
    The input data can be a dictionary, a list or a numpy array.
    The prediction is based on a classification model and two regression models.
    The classification model is a Logistic Regression model that predicts if the diamond is exclusive or common.
    The regression models are LightGBM models that predict the price of the diamond based on its category.
    The prediction is made by first transforming the input data and then using the classification model to determine the category of the diamond.
    The priority is given to the common category using the price threshold to be more accurate in that category.
    The category common is adjusted by inflation rate to predict the price of the diamond in the current year.
    """
    cat_threshold = 0.92
    avg_infl_rate = 0.04
    delta_year = dt.datetime.now().year - 2008
    data = data_transformation(data)

    if len(data) == 1:
        cat = log_reg.predict_proba(data)[:, 1]
        if cat > cat_threshold:
            cat = 'Exclusive'
            pred = lgbm_exclusive.predict(data)[0]
        else:
            cat = 'Common'
            pred = lgbm_common.predict(data)[0]
            pred = pred * (1 + avg_infl_rate) ** delta_year
        return round(pred, 1)

    else:
        cat = log_reg.predict_proba(data)[:, 1]
        cat = np.where(cat > cat_threshold, 'Exclusive', 'Common')
        pred = np.where(cat == 'Exclusive', lgbm_exclusive.predict(
            data), lgbm_common.predict(data))
        pred = np.where(cat == 'Common', pred *
                        (1 + avg_infl_rate) ** delta_year, pred)
        return pred.round(1)
