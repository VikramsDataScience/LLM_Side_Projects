import os
import sys
import contextlib
import pandas as pd
from missforest.missforest import MissForest
from pathlib import Path
import yaml

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

data_path = Path(global_vars['data_path'])

@contextlib.contextmanager
def suppress_stdout():
    """
    For any library that contains (undesirably) verbose output. Use this boilerplate function to suppress.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Define columns for casting and interval definitions
df = pd.read_excel(data_path / 'ECommerce_Dataset.xlsx', sheet_name=1)
float_columns = ['Tenure', 'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
categorical_columns = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'Complain', 'Churn', 'CouponUsed']
onehot_categoricals = ['Tenure', 'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
missforest_imputer = MissForest()

# Cast float_columns as integers, impute NaN values using MissForest
with suppress_stdout():
    df = missforest_imputer.fit_transform(x=df,
                                    categorical=categorical_columns)
    df[float_columns] = df[float_columns].astype(int)

# Create bins for 'Tenure': '0–12 Month', '12–24 Month', '24–48 Months', '48–60 Month', '> 60 Month'
df['Tenure'] = pd.cut(df['Tenure'], [0, 12, 24, 48, 60, 72], right=False) # Set arg 'right=False' to close the left interval (i.e. 0 <= x < 12 in the 'Tenure' column). Otherwise, NaNs will occur!
print(df.value_counts('Tenure'))

df.set_index('CustomerID', inplace=True)
df = pd.get_dummies(df, columns=onehot_categoricals, dtype=int)
# Rename the closed interval '[' columns to suit XGBClassifier() class. Otherwise XGBClassifier() will raise column name errors
df.columns = [col.replace('[', '(') for col in df.columns]
print(df.info())
print(df)

# Save PreProcessed Data Frame for downstream consumption
df.to_csv(data_path / 'PreProcessed_ECommerce_Dataset.csv')