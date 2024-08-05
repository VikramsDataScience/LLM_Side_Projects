import pandas as pd
from pathlib import Path
from .config import Config

# Load the file paths and global variables from the Config file
config = Config()
data_path = config.data_path
onehot_categoricals = config.onehot_cols

def pre_processing(df_path, bins, onehot_cols, output_path, bin_cols=str, sheet_name=None):
    """
    Positional arg definitions:
    - bins: Set range of bins for pd.cut() to use. Must be List data structure
    - bin_cols: Which col names need to parsed. Must be 'str' dtype
    - sheet_name (OPTIONAL): Only specify when reading Excel files to indicate which Excel sheet to read
    into the DataFrame.
    """
    if '.xlsx' in df_path.suffix:
        df = pd.read_excel(df_path, sheet_name)
    elif '.csv' in df.suffix:
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

# Create bins for 'Tenure': '0–12 Month', '12–24 Month', '24–48 Months', '48–60 Month', '> 60 Month'
df = pre_processing(df_path=Path(data_path) / 'ECommerce_Dataset_IMPUTED.csv', bins=[0, 12, 24, 48, 60, 72], onehot_cols=onehot_categoricals, output_path=Path(data_path) / 'PreProcessed_ECommerce_Dataset.csv', bin_cols='Tenure')
print(df.info())
print(df)