from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import scipy.sparse as sp
import torch
import sys
import yaml

# Load the file paths and global variables from YAML config file
config_path = Path('C:/Users/Vikram Pande/Side_Projects/Recommender_Model/src')

with open(config_path / 'config.yml', 'r') as file:
    global_vars = yaml.safe_load(file)

# Import NCF_Architecture module
sys.path.append(r'C:\Users\Vikram Pande\Side_Projects\Recommender_Model\src')
from NCF_Architecture_config import NCF

# Declare global variables from loaded YAML file
model_ver = global_vars['model_ver']
embedding_dim = global_vars['embedding_dim']
files_path = Path(global_vars['files_path'])
product_lookup = pd.read_csv(files_path / 'products.csv')
sp_matrix = sp.load_npz(files_path / f'sparse_matrix_{model_ver}.npz').tocoo() # Convert to 'coordinate sparse matrix' as prep for the Tensor conversion

# If GPU is available, instantiate a device variable to use the GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Get the number of orders and products from the sparse matrix
num_orders = sp_matrix.shape[0]
num_products = sp_matrix.shape[1]

# Load pretrained model and mount to GPU
model = NCF(num_orders, num_products, embedding_dim)
model.load_state_dict(torch.load(files_path / f'model_state_{model_ver}.pth'))
model.to(device)

# Commence evaluation step
def evaluate(model, ):