from transformers import GPT2LMHeadModel
import tiktoken
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import logging
from pathlib import Path

logger = logging.getLogger('LLM_TrainTokenize')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/LLM_TrainTokenize_log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects/Generative_AI_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

content_file = global_vars['content_file']
pretrained_HG_model = global_vars['pretrained_HG_model']
LLM_pretrained_path = global_vars['LLM_pretrained_path']
batch_size = global_vars['batch_size']
model_ver = global_vars['model_ver']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pretrained model from HuggingFace Hub
pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_HG_model).to(device)

# Load custom text corpus from the TXT file created by the 'LLM_PreProcessing' module
# IMPORTANT N.B.: If using OpenAI's 'tiktoken', the TXT file must be 'utf-8' encoded to ensure lossless execution. Otherwise tiktoken will generate lossy (data will leak) tokens during the encode/decode steps
with open(Path(content_file), 'r', encoding='utf-8') as f:
    custom_text = f.read()

# Use OpenAI's GPT-4 Byte Pair Encoding (BPE) tokenizer to encode the Job ad text corpus
tokenizer = tiktoken.get_encoding('cl100k_base')
# 'allow_special' arg refers to correctly tokenizing any special characters that may exist. It can be set to 'all', 'none', or a custom list of special tokens to allow/disallow
print('Encoding custom TXT file...')
encoded_text = tokenizer.encode(custom_text, allowed_special='all')
print('Encoding successfully completed!')

# Split into Train & Test sets and convert to Tensor data structure
train_data, test_data = train_test_split(encoded_text, test_size=0.10, random_state=314)
print('Saving test set as:', f'LLM_test_data_{model_ver}.npy')

np.save(Path(LLM_pretrained_path) / f'LLM_test_data_{model_ver}.npy', test_data)
print('Testing set saved successfully! \nTesting set storage location:', Path(LLM_pretrained_path) / f'LLM_test_data_{model_ver}.npy')
# Delete 'test_data' since it's been saved to disk for validation further downstream
del test_data

train_tensor = torch.LongTensor(train_data).to(device)
# Divide training set into batches for the training loop
train_batches = DataLoader(train_tensor, batch_size=batch_size)

print(f'No. of parameters in the Pretrained \'{pretrained_HG_model}\' model: {pretrained_model.num_parameters():,}')
