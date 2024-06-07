import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

content_file = global_vars['content_path']
data_path = global_vars['data_path']

# Define columns for casting and interval definitions
df = pd.read_excel(Path(content_file), sheet_name=1)
float_columns = ['Tenure', 'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

# Cast float_columns as integers, impute NaN values with 0s, and create bins for 'Tenure': '0–12 Month', '12–24 Month', '24–48 Months', '48–60 Month', '> 60 Month'
df[float_columns] = df[float_columns].fillna(0).astype(int)
print('\nRECASTED DATA FRAME WITHOUT NaN VALUES:\n', df)
df['Tenure'] = pd.cut(df['Tenure'], [0, 12, 24, 48, 60, 72])
print(df.value_counts('Tenure'))

