import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # Use this if using N-Grams for feature extraction
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

# Define file paths, load the necesssary CSV files, and declare global variables
files_path = Path('C:/Sample Data/Recommender_data')
product_lookup = pd.read_csv(files_path / 'products.csv')
embedding_size = 50
batch_size = 32

# If GPU is available, instantiate a device variable to use the GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load and inspect Preprocessed data left by the upstream 'Preprocessing_EDA' component
sparse_df = pd.read_parquet(files_path / 'sparse_df.gz')
sp_matrix = sp.load_npz(files_path / 'sparse_matrix_v0.0.1.npz')
print('SPARSE MATRIX:' + '\n', sp_matrix)
print('SPARSE DF:' + '\n', sparse_df)

# Generate train/test sets with a reproducable seed for inference
train_sparse, test_sparse = train_test_split(sp_matrix, test_size=0.30, random_state=1)

n_ids = sparse_df['order_id'].nunique()
n_products = sparse_df['product_id'].nunique()

# Mount to GPU
#model.to(device)