from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import torch
from pathlib import Path
import logging
import yaml

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

LLM_pretrained_path = global_vars['LLM_pretrained_path']
training_log_path = global_vars['training_log_path']
batch_size = global_vars['batch_size']
pretrained_model = global_vars['pretrained_HG_model']
content_file_path = global_vars['content_file']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)

custom_text = load_dataset('text', data_files='C:/Sample Data/Job_Ad_QA_data/content_cleaned.txt')

def encode_batches(batch):
    tokenized_data = tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors='pt')
    return tokenized_data

tokenized_data = custom_text.map(encode_batches, batched=True, batch_size=batch_size)
tokenized_data.save_to_disk(Path(LLM_pretrained_path))