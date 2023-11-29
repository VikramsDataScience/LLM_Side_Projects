from os import path
import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport

# Define file paths
files_path = Path('C:/Sample Data')
preprocess_path = Path('C:/Users/Vikram Pande/LLM_Side_Projects/Recommender_Model/src/Preprocessing_EDA')

# Read in the individual csv files, and store as separate Pandas Dataframes
aisles = pd.read_csv(files_path / 'aisles.csv')
departments = pd.read_csv(files_path / 'departments.csv')
order_products__prior = pd.read_csv(files_path / 'order_products__prior.csv')
order_products__train = pd.read_csv(files_path / 'order_products__train.csv')
products = pd.read_csv(files_path / 'products.csv')

# Merge data sets as a left join using 'product_id', 'department_id', and 'aisle_id' as join predicates
merged_products = pd.merge(products, aisles, on='aisle_id', how='left')
merged_products = pd.merge(merged_products, departments, on='department_id', how='left')
merged_products = pd.merge(merged_products, order_products__train, on='product_id', how='left')
merged_products = merged_products.reindex(columns=['order_id','product_id','product_name','aisle_id','aisle','department_id','department', 'add_to_cart_order', 'reordered'])
print(merged_products)

# If the EDA report doesn't exist, conduct EDA by generating a ydata Profile Report
if not path.exists(preprocess_path / 'EDA_Profile_Report.html'):
    profile_report = ProfileReport(merged_products, tsmode=False, explorative=True, dark_mode=True)
    profile_report.to_file(preprocess_path / 'EDA_Profile_Report.html')

# If there are missing values, assess where these values exist
print(merged_products.isna().sum())

# Save Preprocessed data into a compressed Parquet DF for downstream consumption
merged_products.to_parquet(files_path / 'merged_products.parquet.gz', index=False, compression='gzip')