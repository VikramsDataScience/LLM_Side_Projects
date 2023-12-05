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
order_products_train = pd.read_csv(files_path / 'order_products__train.csv')
products = pd.read_csv(files_path / 'products.csv')
order_products_prior = pd.read_csv(files_path / 'order_products__prior.csv') # Read in historical product purchases

# Merge data sets as a left join using 'product_id', 'department_id', and 'aisle_id' as join predicates
merged_products = pd.merge(products, aisles, on='aisle_id', how='left')
merged_products = pd.merge(merged_products, departments, on='department_id', how='left')
merged_products = pd.merge(merged_products, order_products_train, on='product_id', how='left')
merged_products = merged_products.reindex(columns=['order_id','product_id','product_name','aisle_id','aisle','department_id','department', 'add_to_cart_order', 'reordered'])

print('FINAL DF:' + '\n', merged_products)

# Prepare 'order_products_prior' DF for generating the target_values that'll be required further downstream
# IMPORTANT N.B.: For Neural Collaborative Filter to work, the 'target_values' must have the same shape as the 'final_df'. Please inspect both frames to verify that they're identically shaped
order_products_prior = order_products_prior.groupby('order_id').sum().reset_index()
target_values = order_products_prior.join(merged_products, on='order_id', how='left', rsuffix='r')
# Remove the duplicated 'rsuffix' columns created by the join() and reorder the columns to match the 'merged_products' DF
target_values = target_values.drop(columns=['order_idr','reorderedr'])
target_values = target_values.reindex(columns=['order_id','product_id','product_name','aisle_id','aisle','department_id','department', 'add_to_cart_order', 'reordered'])
print('TARGET VALUES DF:' + '\n', target_values)

# If the EDA report doesn't exist, conduct EDA by generating a ydata Profile Report
if not path.exists(preprocess_path / 'EDA_Profile_Report.html'):
    profile_report = ProfileReport(merged_products, tsmode=False, explorative=True, dark_mode=True)
    profile_report.to_file(preprocess_path / 'EDA_Profile_Report.html')

# Save Preprocessed data into a compressed Parquet DFs for downstream consumption
# merged_products.to_parquet(files_path / 'merged_products_train.parquet.gz', index=False, compression='gzip')
# target_values.to_parquet(files_path / 'target_values.parquet.gz', index=False, compression='gzip')