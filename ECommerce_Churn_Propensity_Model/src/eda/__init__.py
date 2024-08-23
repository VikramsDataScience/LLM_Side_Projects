from ..config import Config
from missforest.missforest import MissForest
from scipy.stats import skew
import numpy as np
import pandas as pd
import os
import sys
import contextlib

# Initialise global variables from config module
config = Config()

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

def read_impute_data(df_path, float_cols, categorical_cols, output_path, sheet_name=None) -> pd.DataFrame:
    """
    Read in Excel/CSV file, define columns for casting & interval definitions, and perform imputation
    with Missing Forest.
    IMPORTANT NOTE: When parsing to the 'df_path' arg, only parse from the 'Path' class in the 'pathlib'
    library.
    - 'sheet_name' (OPTIONAL): Only required when reading Excel files to indicate which Excel sheet to read
    into the DataFrame.
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