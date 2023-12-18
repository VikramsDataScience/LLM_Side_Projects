from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

# Load the file paths and global variables from YAML config file
config_path = Path('C:/Users/Vikram Pande/Side_Projects/Recommender_Model/src')

with open(config_path / 'config.yml', 'r') as file:
    global_vars = yaml.safe_load(file)

# Import NCF_Architecture module
sys.path.append(global_vars['NCF_path'])
from NCF_Architecture_config import NCF

# Declare global variables from config YAML file
files_path = Path(global_vars['files_path'])
model_ver = global_vars['model_ver']
embedding_dim = global_vars['embedding_dim']
batch_size = global_vars['batch_size']
num_epochs = global_vars['num_epochs']
step_size = global_vars['step_size']
gamma = global_vars['gamma']

# If GPU is available, instantiate a device variable to use the GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load and inspect Preprocessed data left by the upstream 'Preprocessing_EDA' component
sp_matrix = sp.load_npz(files_path / f'sparse_matrix_{model_ver}.npz').tocoo() # Specify coordinate sparse matrix as prep for the Tensor conversion
print('SPARSE MATRIX:' + '\n', sp_matrix)

# Extract and assign 'order_id', 'product_id', and 'reordered' values from the sparse matrix
order_id = sp_matrix.row
product_id = sp_matrix.col
reordered = sp_matrix.data # 'reordered' flag is surrigate indicator of ratings in a traditional Recommender
print('order_id:' + '\n', order_id)
print('product_id:' + '\n', product_id)
print('reordered:' + '\n', reordered)

# Generate train/test sets with a reproducable seed for inference
(order_train, order_test, 
 product_train, product_test, 
 reordered_train, reordered_test) = train_test_split(order_id, product_id, reordered, 
                                                     test_size=0.30, 
                                                     random_state=1)

# Convert 'train' sets to Tensor data structure
order_train_tensor = torch.LongTensor(order_train).to(device)
product_train_tensor = torch.LongTensor(product_train).to(device)
reordered_train_tensor = torch.FloatTensor(reordered_train).to(device)

# Save test sets to disk for the downstream 'NCF_Model_Evaluate' module
np.save(files_path / f'order_test_{model_ver}.npy', order_test)
np.save(files_path / f'product_test_{model_ver}.npy', product_test)
np.save(files_path / f'reordered_test_{model_ver}.npy', reordered_test)
# Delete test sets from module to save memory
del order_test, product_test, reordered_test

# Get the number of orders and products from the sparse matrix
num_orders = sp_matrix.shape[0]
num_products = sp_matrix.shape[1]

# Instantiate the model, loss function, scheduler, scaler and optimizer for back propogation
model = NCF(num_orders, num_products, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 'step_size' refers to adjusting learning rate by the number of epochs multiplied by 'gamma' (i.e. step_size=2, gamma=0.1 means changing LR every 2 x 0.1 = 0.2 epochs)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
scaler = GradScaler()

# Mount the model and training tensors to the GPU
model.to(device)

# Load and train the model
train_dataset = TensorDataset(order_train_tensor,
                              product_train_tensor,
                              reordered_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    model.train()
    for batch_orders, batch_products, batch_reorders in tqdm(train_loader, desc='Training Steps'):
        optimizer.zero_grad()

        # Use Mixed Precision Testing to improve training time
        with autocast():
            outputs = model(batch_orders, batch_products)
            loss = criterion(outputs, batch_reorders)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # LR scheduler
    scheduler.step()

# Save trained model to disk location
torch.save(model.state_dict(), files_path / f'model_state_{model_ver}.pth')