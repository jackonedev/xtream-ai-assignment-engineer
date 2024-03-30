import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=42, verbose=False):
    
    """
    Custom function to split the data into training, validation and testing sets
    
    Parameters:
    X: pd.DataFrame or np.ndarray
        Features matrix
    y: pd.DataFrame or np.ndarray
        Target variable
    val_size: float
        Proportion of the data to include in the validation set
    test_size: float
        Proportion of the data to include in the testing set
    random_state: int
        Random seed
    verbose: bool
        Print information about the data split
    
    Returns:
    X_train: pd.DataFrame or np.ndarray
        Training features matrix
    X_val: pd.DataFrame or np.ndarray
        Validation features matrix
    X_test: pd.DataFrame or np.ndarray
        Testing features matrix
    y_train: pd.DataFrame or np.ndarray
        Training target variable
    y_val: pd.DataFrame or np.ndarray
        Validation target variable
    y_test: pd.DataFrame or np.ndarray
        Testing target variable
    """

    skf = KFold(n_splits=2, shuffle=True, random_state=random_state)

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        for train_index, test_index in skf.split(X, y):
            X_1, X_2 = X.iloc[train_index], X.iloc[test_index]
            y_1, y_2 = y.iloc[train_index], y.iloc[test_index]

    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        for train_index, test_index in skf.split(X, y):
            X_1, X_2 = X[train_index], X[test_index]
            y_1, y_2 = y[train_index], y[test_index]
    
    m = X.shape[0]
    training_size = int(m * (1 - test_size) )
    testing_size = m - training_size
    validation_size = int(training_size * val_size)
    training_size -= validation_size
    
    if verbose:
        msg = """DATA SPLIT INFORMATION:\
        \nTotal rows: {} - columns: {}\
        \nTraining rows: {}\
        \nValidation rows: {}\
        \nTesting rows: {}\
        """.format(
            m, X.shape[1], training_size, validation_size, testing_size
        )
        print(msg)

    X_train_1, X_val, y_train_1, y_val = train_test_split(
        X_1, y_1, test_size=validation_size, random_state=random_state
    )
    X_train_2, X_test, y_train_2, y_test = train_test_split(
        X_2, y_2, test_size=testing_size, random_state=random_state
    )

    X_train = pd.concat([pd.DataFrame(X_train_1), pd.DataFrame(X_train_2)])
    y_train = pd.concat([pd.DataFrame(y_train_1), pd.DataFrame(y_train_2)])

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame)
        assert isinstance(y_val, pd.DataFrame)
        assert isinstance(y_test, pd.DataFrame)
        return X_train, X_val, X_test, y_train, y_val, y_test
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X_train = X_train.values
        y_train = y_train.values
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
def build_features(one_hot_encoder, dataset, categorical_features):
    """
    Return the one-hot encoded features in a DataFrame
    
    Parameters:
    one_hot_encoder: OneHotEncoder
        OneHotEncoder object fitted
    dataset: pd.DataFrame
        Dataset to encode
    categorial_features: list
        List of categorial features to encode
        
    Returns:
    result: pd.DataFrame
        DataFrame with the one-hot encoded features
    """
    encoded_features = one_hot_encoder.transform(
        dataset[categorical_features]
    )
    result = pd.DataFrame()
    last_len = 0
    for i, features in enumerate(one_hot_encoder.categories_):
        len_feature = len(features)
        formated_features = [
            f"{dataset[categorical_features].columns[i]}_{feat}".replace(
                " ", "_"
            )
            for feat in features
        ]
        builded_features = pd.DataFrame(
            encoded_features[:, last_len : last_len + len_feature],
            columns=formated_features,
        )
        result = pd.concat([result, builded_features], axis=1)
        last_len += len_feature
    return result
    
    
    