from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import yaml
import logging

logger = logging.getLogger('NCF_Model_Evaluate')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/NCF_Model_Evaluate_Log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects/Recommender_Model/src')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

# Import NCF_Architecture module
sys.path.append(global_vars['NCF_path'])
from NCF_Architecture_config import NCF

# Declare global variables from config YAML file
model_ver = global_vars['model_ver']
embedding_dim = global_vars['embedding_dim']
batch_size = global_vars['batch_size']
total_loss = global_vars['total_loss']
files_path = Path(global_vars['files_path'])
trained_models = Path(global_vars['trained_models'])

# Load saved data left by upstream modules
product_lookup = pd.read_csv(files_path / 'products.csv')
sp_matrix = sp.load_npz(trained_models / f'sparse_matrix_{model_ver}.npz').tocoo() # Convert to 'coordinate sparse matrix' as prep for the Tensor conversion
order_test = np.load(trained_models / f'order_test_{model_ver}.npy')
product_test = np.load(trained_models / f'product_test_{model_ver}.npy')
reordered_test = np.load(trained_models / f'reordered_test_{model_ver}.npy')
model_state = torch.load(trained_models / f'model_state_{model_ver}.pth')

# If GPU is available, instantiate a device variable to use the GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Get the number of orders and products from the sparse matrix
num_orders = sp_matrix.shape[0]
num_products = sp_matrix.shape[1]

# Load pretrained model and mount to GPU
model = NCF(num_orders, num_products, embedding_dim)
model.load_state_dict(torch.load(trained_models / f'model_state_{model_ver}.pth'))
model.to(device)

# Extract and assign 'order_id', 'product_id', and 'reordered' values from the sparse matrix
order_id = sp_matrix.row
product_id = sp_matrix.col
reordered = sp_matrix.data # 'reordered' flag is surrigate indicator of ratings in a traditional Recommender
print('order_id:' + '\n', order_id)
print('product_id:' + '\n', product_id)
print('reordered:' + '\n', reordered)

# Convert the 'test sets' to Tensor data structure and mount to the GPU
order_test_tensor = torch.LongTensor(order_test).to(device)
product_test_tensor = torch.LongTensor(product_test).to(device)
reordered_test_tensor = torch.FloatTensor(reordered_test).to(device)

# Load the Test data
criterion = nn.MSELoss()
test_dataset = TensorDataset(order_test_tensor, 
                             product_test_tensor, 
                             reordered_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Commence evaluation loop
model.eval()
total_loss = 0.0
predictions = []
targets = []

with torch.no_grad():
    for batch_orders, batch_products, batch_reorders in tqdm(test_loader, desc='Model Evaluation Progress'):
        outputs = model(batch_orders, batch_products)
        loss = criterion(outputs, batch_reorders)
        total_loss += loss.item()

        predictions.extend(outputs.cpu().numpy())
        targets.extend(batch_reorders.cpu().numpy())

# Calculate & save evaluation metrics to storage
average_loss = total_loss / len(test_loader)
mae = mean_absolute_error(targets, predictions)
rmse = np.sqrt(mean_squared_error(targets, predictions))

print('AVERAGE LOSS: ', average_loss)
print('MEAN ABSOLUTE ERROR: ', mae)
print('ROOT MEAN SQUARED ERROR: ', rmse)

with open(trained_models / 'evaluation_metrics.txt', 'w') as eval_metrics:
    eval_metrics.write(f'AVERAGE LOSS: {average_loss}\n')
    eval_metrics.write(f'MEAN ABSOLUTE ERROR: {mae}\n')
    eval_metrics.write(f'ROOT MEAN SQUARED ERROR: {rmse}\n')