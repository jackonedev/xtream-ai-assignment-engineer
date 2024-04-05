from typing import Any, Callable
import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from pydantic import ValidationError

from schemas.diamond import Diamond, DiamondDataFrame
from utils.config import MODELS_PATH, DATASET_PATH


# LOAD THE MODELS AND TRANSFORM PIPELINE

with open(os.path.join(MODELS_PATH, "log_reg_pipeline.pkl"), "rb") as file:
    log_reg_pipeline = pickle.load(file)

with open(os.path.join(MODELS_PATH, "common_pipeline.pkl"), "rb") as file:
    common_pipeline = pickle.load(file)

with open(os.path.join(MODELS_PATH, "exclusive_pipeline.pkl"), "rb") as file:
    exclusive_pipeline = pickle.load(file)

with open(os.path.join(MODELS_PATH, "log_reg.pkl"), "rb") as file:
    log_reg = pickle.load(file)

with open(os.path.join(MODELS_PATH, "lgbm_common.pkl"), "rb") as file:
    lgbm_common = pickle.load(file)

with open(os.path.join(MODELS_PATH, "lgbm_exclusive.pkl"), "rb") as file:
    lgbm_exclusive = pickle.load(file)


def data_transformation(data: Any) -> pd.DataFrame:
    cols = pd.read_csv(DATASET_PATH).drop(columns='price').columns

    if isinstance(data, dict):
        try:
            # Validation block
            try:
                Diamond(**data)
            except:
                conv = pd.DataFrame(data).to_dict(orient='records')
                DiamondDataFrame(data=conv)
            
        except ValidationError as e:
            print("Error transforming the input:")
            for element in e.errors():
                element.pop('input')
                element.pop('url')
                print(element)
            raise e

        data = pd.DataFrame(data, index=[0])

    elif isinstance(data, type(np.array(0))) or isinstance(data, list):
        data = {col: [val] for col, val in zip(cols, data)}
        data = pd.DataFrame(data)
        
    elif isinstance(data, pd.Series):
        data = data.to_frame().T

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
    
    _data = data_transformation(data)
    _X = log_reg_pipeline.transform(_data)

    if len(_X) == 1:
        # Single prediction
        cat = log_reg.predict_proba(_X)[:, 1]
        if cat > cat_threshold:
            cat = 'Exclusive'
            _X = exclusive_pipeline.transform(_data)
            pred = lgbm_exclusive.predict(_X)[0]
        else:
            cat = 'Common'
            _X = common_pipeline.transform(_data)
            pred = lgbm_common.predict(_X)[0]
            pred = pred * (1 + avg_infl_rate) ** delta_year
        return round(pred, 1)

    else:
        # Batch prediction
        cat = log_reg.predict_proba(_X)[:, 1]
        cat = np.where(cat > cat_threshold, 'Exclusive', 'Common')
        
        original_order = {}
        for i, z in enumerate(zip(_data.index, cat)):
            original_order[i] = tuple(z)
            
        agrupated_index = {'Exclusive': [], 'Common': []}
        for oo in original_order.values():
            if oo[1] == 'Exclusive':
                agrupated_index['Exclusive'].append(oo[0])
            else:
                agrupated_index['Common'].append(oo[0])
                
        _Xe = exclusive_pipeline.transform(_data.loc[agrupated_index['Exclusive']])
        _Xc = common_pipeline.transform(_data.loc[agrupated_index['Common']])

        pred_exclusive = lgbm_exclusive.predict(_Xe)
        pred_common = lgbm_common.predict(_Xc)
        pred_common = pred_common * (1 + avg_infl_rate) ** delta_year
                
        pred = np.concatenate([pred_exclusive, pred_common])
        pred = pred[np.argsort(list(original_order.keys()))]
        
        return pred.round(1)
