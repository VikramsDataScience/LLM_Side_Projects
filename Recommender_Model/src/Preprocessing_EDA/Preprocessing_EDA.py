from os import path
import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

# Define file paths and any required global variables
files_path = Path('C:/Sample Data')
preprocess_path = Path('C:/Users/Vikram Pande/LLM_Side_Projects/Recommender_Model/src/Preprocessing_EDA')
sparse_df = pd.DataFrame()

# Read in the individual csv files, and store as separate Pandas Dataframes
aisles = pd.read_csv(files_path / 'aisles.csv')
departments = pd.read_csv(files_path / 'departments.csv')
order_products_train = pd.read_csv(files_path / 'order_products__train.csv')
products = pd.read_csv(files_path / 'products.csv')
orders = pd.read_csv(files_path / 'order_products__prior.csv') # Read in historical product purchases
print('*****PREVIOUS ORDERS:*****' + '\n', orders)

# Merge data sets as a left join using 'product_id', 'department_id', and 'aisle_id' as join predicates
merged_products = pd.merge(products, aisles, on='aisle_id', how='left')
merged_products = pd.merge(merged_products, departments, on='department_id', how='left')
merged_products = pd.merge(merged_products, order_products_train, on='product_id', how='left')
merged_products = merged_products.reindex(columns=['order_id','product_id','product_name','aisle_id','aisle','department_id','department', 'add_to_cart_order', 'reordered'])
print('*****FINAL DF:*****' + '\n', merged_products)

# Define function for generating a sparse matrix
def sparse_matrix(order_ids, item_ids):
    sparse_df['order_id'] = orders['order_id'].map(order_ids)
    sparse_df['product_id'] = merged_products['product_id'].map(item_ids)
    # Replace NaN values with a placeholder (e.g., 0)
    sparse_df['order_id'] = sparse_df['order_id'].fillna(0)
    sparse_df['product_id'] = sparse_df['product_id'].fillna(0)
    # Cast as integers
    sparse_df['order_id'] = sparse_df['order_id'].astype(int)
    sparse_df['product_id'] = sparse_df['product_id'].astype(int)
    print('NUM_ORDERS:' + '\n', sparse_df['order_id'])
    print('NUM_PRODUCTS:'+ '\n', sparse_df['product_id'])
    sparse_matrix = sp.coo_matrix((np.ones(len(sparse_df)), (sparse_df['order_id'], sparse_df['product_id'])))
    return sparse_matrix

# Perform preprocessing for Neural Collaborative Filtering
order_ids = {order_ids: idx for idx, order_ids in enumerate(orders['order_id'].unique())}
item_ids = {item_ids: idx for idx, item_ids in enumerate(merged_products['product_id'].unique())}
sparse_matrix(order_ids, item_ids)

# If the EDA report doesn't exist, conduct EDA by generating a ydata Profile Report
if not path.exists(preprocess_path / 'EDA_Profile_Report.html'):
    profile_report = ProfileReport(merged_products, tsmode=False, explorative=True, dark_mode=True)
    profile_report.to_file(preprocess_path / 'EDA_Profile_Report.html')

# Save Preprocessed data into a compressed Parquet DFs for downstream consumption
# merged_products.to_parquet(files_path / 'merged_products_train.parquet.gz', index=False, compression='gzip')
# orders.to_parquet(files_path / 'target_values.parquet.gz', index=False, compression='gzip')