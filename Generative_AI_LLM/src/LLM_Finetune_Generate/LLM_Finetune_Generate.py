from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset, Dataset
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
model_output_path = global_vars['model_output_path']
pretrained_model = global_vars['pretrained_HG_model']
content_file_path = global_vars['content_file']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)

print('Loading Tokenized Batches from disk location...')
tokenized_batches = Dataset.load_from_disk(Path(LLM_pretrained_path)/ 'train')
print(tokenized_batches)

# output = model.generate(tokenized_data, 
#                         max_length=1024, 
#                         do_sample=True, 
#                         top_k=50, 
#                         top_p=0.95, 
#                         num_return_sequences=1)

# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)