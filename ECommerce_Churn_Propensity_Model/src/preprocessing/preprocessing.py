from pathlib import Path
# Load variables from __init__.py
from . import pre_processing, Config

# Load the file paths and global variables from the Config file
config = Config()
data_path = config.data_path
onehot_categoricals = config.onehot_cols

# Create bins for 'Tenure': '0–12 Month', '12–24 Month', '24–48 Months', '48–60 Month', '> 60 Month'
df = pre_processing(df_path=Path(data_path) / 'ECommerce_Dataset_IMPUTED.csv', 
                    bins=[0, 12, 24, 48, 60, 72], 
                    onehot_cols=onehot_categoricals, 
                    output_path=Path(data_path) / 'PreProcessed_ECommerce_Dataset.csv', 
                    bin_cols='Tenure')
print(df.info())
print(df)