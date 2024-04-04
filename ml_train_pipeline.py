import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from pipelines.estimators import (
    RemoveOutliers,
    CustomOrdinalEncoder,
    CustomOneHotEncoder,
    CustomScaler
)
from pipelines.column_transformation import PandasFeatureUnion
from pipelines.functions import column_categorization, select_numerical_features
from tools.preprocessing import train_val_test_split
from utils.config import (
    DATASET_PATH, SEED,
    MODELS_PATH, MODEL_DATA_PATH
)


##  MAIN PROGRAM  ##

df = pd.read_csv(DATASET_PATH)
df = RemoveOutliers().fit_transform(df)
column_categories = column_categorization(df.drop(columns=["price"]))


# 1- Classification Model
price_threshold = df.price.median()
X = df.drop(columns=["price"])
y_cat = pd.DataFrame(
    np.where(df.price > price_threshold, 1, 0), columns=['exclusive'])

X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

workflow_1 = PandasFeatureUnion([
    ("numeric_cols", FunctionTransformer(select_numerical_features)),
    ("binary_cols", CustomOrdinalEncoder(
        column_categories["binary_features"])),
    ("category_cols", CustomOneHotEncoder(
        column_categories["categorical_features"])),
])

pipeline_1 = Pipeline([
    ("column_transformation", workflow_1),
    ("scale", CustomScaler(StandardScaler)),
])


# 1.1- Train the model
pipeline_1.fit(X_train)
X_train = pipeline_1.transform(X_train)
X_test = pipeline_1.transform(X_test)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_cat_train)

# 1.2- Save the pipeline
with open(os.path.join(MODELS_PATH, "transform_pipeline.pkl"), "wb") as f:
    pickle.dump(pipeline_1, f)


# 2- Regression Models
common_sample = df.loc[np.where(df.price <= price_threshold)[0], :]
exclusive_sample = df.loc[np.where(df.price > price_threshold)[0], :]

params = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.01,
    "max_depth": 10,
    "num_leaves": 14,
    "min_child_samples": 8,
    "verbose": 0,
    "n_jobs": -1,
    "extra_trees": False,
    "random_state": SEED,
}


# 2.1- Common Model
X_common = common_sample.drop(columns=["price"])
y_common = common_sample[["price"]]
Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = train_val_test_split(
    X_common, y_common,
    test_size=0.002,
    random_state=SEED,
    verbose=True
)

pipeline_2 = Pipeline([
    ("column_transformation", workflow_1),
    ("scale", CustomScaler(StandardScaler)),
])

pipeline_2.fit(Xc_train)
Xc_train = pipeline_2.transform(Xc_train)
Xc_val = pipeline_2.transform(Xc_val)
Xc_test = pipeline_2.transform(Xc_test)

# 2.1.1- Train the model
train_set = lgb.Dataset(Xc_train, label=yc_train)
val_set = lgb.Dataset(Xc_val, label=yc_val)
lgbm_common = lgb.train(params, train_set, num_boost_round=3000, callbacks=[
                        lgb.early_stopping(100)], valid_sets=[train_set, val_set])


# 2.2- Exclusive Model
params = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.005,
    "max_depth": 14,
    "num_leaves": 25, 
    "min_child_samples": 1,
    "verbose": 0,
    "n_jobs": -1,
    "extra_trees": False,
    "random_state": SEED,
}

X_exclusive = exclusive_sample.drop(columns=["price"])
y_exclusive = exclusive_sample[["price"]]

Xe_train, Xe_val, Xe_test, ye_train, ye_val, ye_test = train_val_test_split(
    X_exclusive, y_exclusive,
    test_size=0.002,
    random_state=SEED,
    verbose=True
)

pipeline_3 = Pipeline([
    ("column_transformation", workflow_1),
    ("scale", CustomScaler(StandardScaler)),
])

pipeline_3.fit(Xe_train)
Xe_train = pipeline_3.transform(Xe_train)
Xe_val = pipeline_3.transform(Xe_val)
Xe_test = pipeline_3.transform(Xe_test)

# 2.2.1- Train the model
train_set = lgb.Dataset(Xe_train, label=ye_train)
val_set = lgb.Dataset(Xe_val, label=ye_val)
lgbm_exclusive = lgb.train(params, train_set, num_boost_round=3000, callbacks=[
                           lgb.early_stopping(100)], valid_sets=[train_set, val_set])



# 3- Save the models
with open(os.path.join(MODELS_PATH, "log_reg.pkl"), "wb") as f:
    pickle.dump(log_reg, f)
with open(os.path.join(MODELS_PATH, "lgbm_common.pkl"), "wb") as f:
    pickle.dump(lgbm_common, f)
with open(os.path.join(MODELS_PATH, "lgbm_exclusive.pkl"), "wb") as f:
    pickle.dump(lgbm_exclusive, f)
    
# 3.1- Save the data
X_train.to_csv(os.path.join(MODEL_DATA_PATH, "log_reg_X_train.csv"), index=False)
y_cat_train.to_csv(os.path.join(MODEL_DATA_PATH, "log_reg_y_cat_train.csv"), index=False)
X_test.to_csv(os.path.join(MODEL_DATA_PATH, "log_reg_X_test.csv"), index=False)
y_cat_test.to_csv(os.path.join(MODEL_DATA_PATH, "log_reg_y_cat_test.csv"), index=False)

Xc_train.to_csv(os.path.join(MODEL_DATA_PATH, "common_Xc_train.csv"), index=False)
yc_train.to_csv(os.path.join(MODEL_DATA_PATH, "common_yc_train.csv"), index=False)
Xc_val.to_csv(os.path.join(MODEL_DATA_PATH, "common_Xc_val.csv"), index=False)
yc_val.to_csv(os.path.join(MODEL_DATA_PATH, "common_yc_val.csv"), index=False)
Xc_test.to_csv(os.path.join(MODEL_DATA_PATH, "common_Xc_test.csv"), index=False)
yc_test.to_csv(os.path.join(MODEL_DATA_PATH, "common_yc_test.csv"), index=False)

Xe_train.to_csv(os.path.join(MODEL_DATA_PATH, "exclusive_Xe_train.csv"), index=False)
ye_train.to_csv(os.path.join(MODEL_DATA_PATH, "exclusive_ye_train.csv"), index=False)
Xe_val.to_csv(os.path.join(MODEL_DATA_PATH, "exclusive_Xe_val.csv"), index=False)
ye_val.to_csv(os.path.join(MODEL_DATA_PATH, "exclusive_ye_val.csv"), index=False)
Xe_test.to_csv(os.path.join(MODEL_DATA_PATH, "exclusive_Xe_test.csv"), index=False)
ye_test.to_csv(os.path.join(MODEL_DATA_PATH, "exclusive_ye_test.csv"), index=False)

print("Train pipeline finished.")
