import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score ,mean_absolute_error, max_error

from tools.preprocessing import build_features



def column_categorization(X):
        numerical_features = X.select_dtypes(include='number').columns
        categories_count = (X.loc[:, (X.dtypes == "object").values]
                .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
                .sum())

        binary_features = categories_count[categories_count == 2].index.to_list()
        categorical_features = categories_count[categories_count > 2].index.to_list()
        
        return {
                'numerical_features': numerical_features,
                'binary_features': binary_features,
                'categorical_features': categorical_features
                }


def select_numerical_features(X):
    """Select the numerical features from the dataset."""
    column_categories = column_categorization(X)
    return X[column_categories["numerical_features"]].reset_index(drop=True)

        
def encode_features(X, encoder, categorical_features):
    return pd.concat(
        [
            X.drop(columns=categorical_features).reset_index(),
            build_features(encoder, X, categorical_features)
        ], axis=1).set_index('index')
    
    
def scale_features(X, scaler):
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns.str.replace(
            r"[^\w\s]", "_", regex=True
        ).str.replace("__+", "_", regex=True),
    )
    

def see_results(y_real, y_pred):
    print("(Total price of sample, Predicted total price, Difference)")
    y_real.price.plot.hist(bins=50, alpha=0.5, color='b', title='REAL: Price distribution'),
    pd.DataFrame(y_pred).plot.hist(bins=50, alpha=0.5, color='b', title='Price prediction distribution'),
    print((total := y_real.price.sum()), (pred_total := int(pd.DataFrame(y_pred).sum().values[0])), (pred_total- total))
        

def evaluate_metrics(y_real, y_pred, verbose=True):
    r2 = r2_score(y_real.price, y_pred)
    r2_var = explained_variance_score(y_real.price, y_pred)
    mae = mean_absolute_error(y_real.price, y_pred)
    max_err = max_error(y_real.price, y_pred)
    if verbose:
        print(f"R2 Score: {r2:.2f} with Variance score: {r2_var:.2f}")
        print(f"MAE Score: {mae:.2f}")
        print(f"Max error: {max_err:.2f}")
    return r2, r2_var, mae, max_err