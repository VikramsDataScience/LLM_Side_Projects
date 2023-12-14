from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import yaml

# Define file paths, load the necesssary CSV files, and global variables
config_path = Path('C:/Users/Vikram Pande/Side_Projects/Recommender_Model/src')

with open(config_path / 'config.yml', 'r') as file:
    global_vars = yaml.safe_load(file)

# Import NCF_Architecture module
sys.path.append(r'C:\Users\Vikram Pande\Side_Projects\Recommender_Model\src')
from NCF_Architecture_config import NCF

files_path = Path(global_vars['files_path'])
model_ver = global_vars['model_ver']
embedding_dim = global_vars['embedding_dim']
batch_size = global_vars['batch_size']
num_epochs = global_vars['num_epochs']

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
order_train_tensor = torch.LongTensor(order_train)
product_train_tensor = torch.LongTensor(product_train)
reordered_train_tensor = torch.FloatTensor(reordered_train)

# Convert 'test' sets to Tensor data structure
order_test_tensor = torch.LongTensor(order_test)
product_test_tensor = torch.LongTensor(product_test)
reordered_test_tensor = torch.FloatTensor(reordered_test)

# Get the number of orders and products from the sparse matrix
num_orders = sp_matrix.shape[0]
num_products = sp_matrix.shape[1]

# Instantiate the model, loss function, and optimizer for back propogation
model = NCF(num_orders, num_products, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mount the model and training tensors to the GPU
model.to(device)

(order_train_tensor, 
 product_train_tensor, 
 reordered_train_tensor) = (order_train_tensor.to(device),
                            product_train_tensor.to(device),
                            reordered_train_tensor.to(device))

# Load and train the model
train_dataset = TensorDataset(order_train_tensor,
                              product_train_tensor,
                              reordered_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    model.train()
    for batch_orders, batch_products, batch_reorders in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_orders, batch_products)
        loss = criterion(outputs, batch_reorders)
        loss.backward()
        optimizer.step()

# Save trained model to disk location
torch.save(model.state_dict(), files_path / f'model_state_{model_ver}.pth')