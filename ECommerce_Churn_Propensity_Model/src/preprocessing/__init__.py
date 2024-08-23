from ..config import Config
import pandas as pd

# Initialise global variables from config module
config = Config()

#################### ADDITIONAL REQUIRED FUNCTIONS ####################
def pre_processing(df_path, bins, onehot_cols, output_path, bin_cols=str, sheet_name=None) -> pd.DataFrame:
    """
    Positional arg definitions:
    - 'bins': Set range of bins for pd.cut() to use. Must be List data structure
    - 'bin_cols': Which col names need to parsed. Must be 'str' dtype
    - 'sheet_name' (OPTIONAL): Only required when reading Excel files to indicate which Excel sheet to read
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
    # df = df.drop(['Unnamed: 0'], axis=1)

    # Save PreProcessed Data Frame for downstream consumption
    df.to_csv(output_path, index=False)
    return df