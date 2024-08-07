from .config import Config
from missforest.missforest import MissForest
from scipy.stats import skew
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path
import os
import sys
import contextlib

# Model version
__version__ = '0.01'

# Initialise global variables from config module
config = Config()
data_path = Path(config.data_path)
churn_app_path = Path(config.churn_app_models)
seed = config.seed

insample_scores = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score'])
outofsample_scores = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score'])
df = pd.read_csv(data_path / 'PreProcessed_ECommerce_Dataset.csv')

# Define target variable (y) and features (X)
# The EDA exposed high correlation with the 'CashbackAmount' feature. So remove from X
X = df.drop(['CustomerID', 'Churn', 'CashbackAmount', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
             'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'Tenure_(12, 24)', 'Tenure_(48, 60)',
             'Tenure_(60, 72)', 'PreferredLoginDevice_Computer', 'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone', 'PreferredPaymentMode_CC',
             'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet', 'Gender_Female',
             'Gender_Male', 'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others', 'MaritalStatus_Divorced',
             'MaritalStatus_Married', 'PreferredPaymentMode_Cash on Delivery'], axis=1)
y = df['Churn']

# Define DMatrix and Hyper Parameters for XGBoost
d_matrix = xgb.DMatrix(data=X, 
                       label=y,
                       enable_categorical=True)
params = {
            'objective':'binary:logistic',
            'max_depth': 9,
            # 'alpha': 10, # L1 Regularization on the leaf nodes (larger value means greater regularization) 
            'lambda': 10, # L2 Regularization on the leaf nodes (larger value means greater regularization). L2 is smoother than L1 and tends to better prevent overfitting
            'learning_rate': 0.4,
            'n_estimators':100,
        }

# Perform Train/Test split with Stratification since the class labels, y, is an imbalanced dataset that favours those who didn't churn (i.e. ~83% didn't churn)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=0.80, 
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=seed)

# Define/load Hyper Parameters
models_config = {
    'logistic_regression': LogisticRegression(class_weight='balanced', # In addition to Stratification, this perform's class balancing on model at fit() stage
                                              solver='liblinear', # The Solver refers to the available Gradient Descent Optimization options (i.e. selection of Loss functions)
                                              random_state=seed),
    'RFClassifier': RandomForestClassifier(class_weight=None, # For RFClassifier, setting class_weights='balanced' harms the F1-Score. Leave class_weight=None (default)
                                           n_estimators=200, # ~200 trees improves F1-Score. 200+ is past the point of optimality, and will reduce accuracy
                                           max_depth=None, # Leave max_depth=None so the classifier can grow the trees until all leaves are pure (lowest Gini Impurity) without stopping criterion causing premature terminations
                                           random_state=seed),
    'XGBoost': xgb.XGBClassifier(**params)
}

# Export functions and classes for use in case you wish to use 'from . import *'
__all__ = [
    'read_impute_data', 
    'doanes_formula',
    'pre_processing',
    'models_config', 
    'insample_scores', 
    'outofsample_scores', 
    'X', 
    'X_train', 
    'X_test', 
    'y_train', 
    'y_test'
]

#################### ADDITIONAL REQUIRED FUNCTIONS ####################
def doanes_formula(data, nan_count) -> int:
    """
    To aid in the preparation for the correct binning (FOR SKEWED DATA COLUMNS ONLY) of Intervals prior to the 
    calculation of Phi K Correlation, I've opted to use Doane's Formula to determine the Bin sizes of the intervals.
    Since I couldn't find a Python library for the formula, I've written this implementation of Doane's Formula.
      Please refer to the 'README.md' file (in the EDA folder) for a mathematical explanation of the formula 
    and the data justifications behind selecting Doane's Formula for calculating bin sizes.
      This function will return the bin length as a truncated integer (not rounded!). I elected for numeric truncation
    over rounding, since I saw that numpy's rounding with np.ceil() led to substantial rounding errors (for instance,
    18.1 would be rounded to 19). So, I've opted to truncate and cast as integer rather than have these rounding 
    errors in the calculation of the interval's bin length.
      N.B.: Since Doane's Formula relies on the number of observations in the DF, if there are NaNs in the input DF, 
    please calculate the number of NaNs and deduct that value from 'n' (i.e. use the 'nan_count' arg in the function).
    If there are no NaNs, please set 'nan_count = 0'.
    """
    n = len(data) - nan_count
    g1 = skew(data) - nan_count
    sigma_g1 = np.sqrt((6*(n - 2)) / ((n + 1)*(n + 3)))
    k = 1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1)
    return int(np.trunc(k))

def read_impute_data(df_path, float_cols, categorical_cols, sheet_name=None) -> pd.DataFrame:
    """
    Read in Excel/CSV file, define columns for casting & interval definitions, and perform imputation
    with Missing Forest.
    IMPORTANT NOTE: When parsing to the 'df_path' arg, only parse from the 'Path' class in the 'pathlib'
    library
    """
    missforest_imputer = MissForest()
    if '.xlsx' in df_path.suffix:
        df = pd.read_excel(df_path, sheet_name)
    elif '.csv' in df_path.suffix:
        df = pd.read_csv(df_path)
    
    # Cast float_columns as integers, impute NaN values using MissForest
    with suppress_stdout():
        df = missforest_imputer.fit_transform(x=df,
                                        categorical=categorical_cols)
        df[float_cols] = df[float_cols].astype(int)
    
    # Save to storage for downstream PreProcessing module
    df.to_csv(Path(data_path) / 'ECommerce_Dataset_IMPUTED.csv')  
    return df

def pre_processing(df_path, bins, onehot_cols, output_path, bin_cols=str, sheet_name=None) -> pd.DataFrame:
    """
    Positional arg definitions:
    - bins: Set range of bins for pd.cut() to use. Must be List data structure
    - bin_cols: Which col names need to parsed. Must be 'str' dtype
    - sheet_name (OPTIONAL): Only specify when reading Excel files to indicate which Excel sheet to read
    into the DataFrame.
    """
    if '.xlsx' in df_path.suffix:
        df = pd.read_excel(df_path, sheet_name)
    elif '.csv' in df_path.suffix:
        df = pd.read_csv(df_path)

    # Set arg 'right=False' to close the left interval (i.e. 0 <= x < 12 in the 'Tenure' column). Otherwise, NaNs will occur!
    df[bin_cols] = pd.cut(df[bin_cols], bins, right=False)
    print(df.value_counts(bin_cols))

    df.set_index('CustomerID', inplace=True)
    df = pd.get_dummies(df, columns=onehot_cols, dtype=int)
    # Rename the closed interval '[' columns to suit XGBClassifier() class. Otherwise XGBClassifier() will raise column name errors
    df.columns = [col.replace('[', '(') for col in df.columns]

    # Save PreProcessed Data Frame for downstream consumption
    df.to_csv(output_path)
    return df

@contextlib.contextmanager
def suppress_stdout():
    """
    For any library that contains (undesirably) verbose output, use this boilerplate function to suppress
    that output in the CLI.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout