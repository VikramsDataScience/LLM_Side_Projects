import pandas as pd
from missforest.missforest import MissForest
from pathlib import Path
from .config import Config, suppress_stdout

# Load the file paths and global variables from the Config file
config = Config()
content_path = config.content_path
data_path = Path(config.data_path)
float_columns = config.float_cols
categorical_columns = config.categorical_cols
onehot_categoricals = config.onehot_cols

def read_impute_data(df_path, sheet_name, float_cols, categorical_cols):
    """
    Read in Excel file, define columns for casting & interval definitions, and perform imputation
    with Missing Forest.
    """
    missforest_imputer = MissForest()
    if '.xlsx' in df_path:
        Path(df_path)
        df = pd.read_excel(df_path, sheet_name)
    elif '.csv' in df_path:
        Path(df_path)
        df = pd.read_csv(df_path)

    # Cast float_columns as integers, impute NaN values using MissForest
    with suppress_stdout():
        df = missforest_imputer.fit_transform(x=df,
                                        categorical=categorical_cols)
        df[float_cols] = df[float_cols].astype(int)
    
    return df

def pre_processing(df, bins, onehot_cols, output_path, bin_cols=str):
    """
    Positional arg definitions:
    - bins: Set range of bins for pd.cut() to use. Must be List data structure
    - bin_cols: Which col names need to parsed. Must be 'str' dtype
    """
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

df = read_impute_data(df_path=content_path, sheet_name=1, float_cols=float_columns, categorical_cols=categorical_columns)
# Create bins for 'Tenure': '0–12 Month', '12–24 Month', '24–48 Months', '48–60 Month', '> 60 Month'
df = pre_processing(df=df, bins=[0, 12, 24, 48, 60, 72], onehot_cols=onehot_categoricals, output_path=data_path / 'PreProcessed_ECommerce_Dataset.csv', bin_cols='Tenure')
print(df.info())
print(df)