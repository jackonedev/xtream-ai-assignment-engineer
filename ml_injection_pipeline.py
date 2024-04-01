from typing import Any
import os
import pickle
import numpy as np
import pandas as pd
from pydantic import ValidationError
from schemas.diamond import Diamond
from utils.config import MODELS_PATH


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
    data = transform_pipeline.transform(data)
    return data


def prediction_pipeline(data: Any) -> float:
    
    data = data_transformation(data)

    cat = log_reg.predict(data)[0]
    
    if cat == 1:
        cat = 'Exclusive'
        pred = lgbm_exclusive.predict(data)[0]
    else:
        cat = 'Common'
        pred = lgbm_common.predict(data)[0]

    return round(pred, 1)


def interpretation_pipeline(data: np.array):
    data = pd.DataFrame(
        data,
        columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    ).T
    print(data.head())
    
    pred = prediction_pipeline(data)
    
    print('prediction made')
    
    return pred
