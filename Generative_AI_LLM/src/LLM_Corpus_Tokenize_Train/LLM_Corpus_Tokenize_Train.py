# from transformers import GPT2Tokenizer, GPT2LMHeadModel
import tiktoken
import torch
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
pretrained_model = global_vars['pretrained_model']
LLM_pretrained_path = global_vars['LLM_pretrained_path']
batch_size = global_vars['batch_size']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load custom text corpus from the TXT file created by the 'LLM_PreProcessing' module
# IMPORTANT N.B.: If using OpenAI's tiktoken, the TXT file must be 'utf-8' encoded to ensure lossless execution. Otherwise tiktoken will generate lossy (incorrect) tokens during the encode/decode steps
with open(Path(content_file), 'r', encoding='utf-8') as f:
    custom_text = f.read()

# Use OpenAI's GPT-4 Byte Pair Encoding (BPE) tokenizer to encode the Job ad text corpus
tokenizer = tiktoken.get_encoding('cl100k_base')
# 'allow_special' arg refers to correctly tokenizing any special characters that may exist. It can be set to 'all', 'none', or a custom list of special tokens to allow
encoded_text = tokenizer.encode(custom_text, allowed_special='all')

# Convert to Tensor data structure and define Train & Test sets
data = torch.tensor(encoded_text, dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n] # Take the first 0.9 of the custom set for training
test = data[n:] # Take the remaining 0.1 of the custom set for validation

