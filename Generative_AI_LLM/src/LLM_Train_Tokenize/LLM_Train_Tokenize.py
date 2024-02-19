from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset, Dataset
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

# config = GPT2Config() # If using, update the GPT2LMHeadModel.from_pretrained(pretrained_model, config=config).to('cuda')
pretrained_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)
print(f'Number of Pretrained Model Parameters: {model.num_parameters():,}')

# Load the custom dataset (job ads corpus used in the 'Jod_Ad_Q&A' model)
clean_content = load_dataset('json', data_files=content_file)

# Define tokenization for custom data (if model doesn't train properly, insert Q&A pairs to help supervise in training the model appropriately)
def encode_batch(batch):
    tokenizer = pretrained_tokenizer(batch['content'], padding=True, max_length=1024, truncation=True, return_tensors='pt')

    # Enable inference mode to eliminate randomizing error loss
    with torch.no_grad():
        # Pass tensors to the model (model is already mounted to the GPU)
        model_output = model(**tokenizer, return_dict=False)

    return model_output

# Perform tokenization and model inference on custom data
docs = clean_content.map(encode_batch, batched=True, batch_size=batch_size)
docs.save_to_disk(Path(LLM_pretrained_path) / 'LLM_Trained_Tokenized')