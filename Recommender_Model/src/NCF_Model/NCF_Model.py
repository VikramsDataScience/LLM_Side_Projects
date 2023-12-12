import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define file paths, load the necesssary CSV files, and declare global variables
files_path = Path('C:/Sample Data/Recommender_data')
product_lookup = pd.read_csv(files_path / 'products.csv')
embedding_size = 50
batch_size = 32

# If GPU is available, instantiate a device variable to use the GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load and inspect Preprocessed data left by the upstream 'Preprocessing_EDA' component
sp_matrix = sp.load_npz(files_path / 'sparse_matrix_v0.0.1.npz').tocoo() # Specify coordinate sparse matrix as prep for the Tensor conversion
print('SPARSE MATRIX:' + '\n', sp_matrix)

# Extract and assign 'order_id', 'product_id', and 'reordered' flags from the sparse matrix
order_id = sp_matrix.row
product_id = sp_matrix.col
reordered = sp_matrix.data # Surrigate indicator of ratings in a traditional Recommender
print('order_id:' + '\n', order_id)
print('product_id:' + '\n', product_id)
print('reordered:' + '\n', reordered)

# Generate train/test sets with a reproducable seed for inference
train_sparse, test_sparse = train_test_split(sp_matrix, test_size=0.30, random_state=1)

# Convert to Tensor data structure
#train_order_ids = torch.LongTensor(train_sparse)

# Mount to GPU
#model.to(device)