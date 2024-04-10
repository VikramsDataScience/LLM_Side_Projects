from transformers import GPT2Tokenizer
import re
from os import remove
from datasets import load_dataset
import torch
from pathlib import Path
import logging
import yaml

logger = logging.getLogger('LLM_TrainTokenize')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Error_Logs/LLM_TrainTokenize_log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Generative_AI_LLM/HuggingFace_LLM')
    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

LLM_pretrained_path = global_vars['LLM_pretrained_path']
batch_size = global_vars['batch_size']
pretrained_model = global_vars['pretrained_HG_model']
seed = global_vars['seed']
training_log_path = global_vars['training_log_path']
pretrained_model = global_vars['pretrained_HG_model']
content_file_path = global_vars['content_file']
train_file = global_vars['train_file']
validate_file = global_vars['validate_file']
content_path = global_vars['content_path']

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token

############# SPLIT TEXT FILE INTO 'TRAIN/VALIDATE' SETS #############
with open(Path(content_path) / 'content_cleaned.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Perform train/validate split with seed for reproduceability
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

torch.manual_seed(seed)
train_set = text_data[:split_idx]
validate_set = text_data[split_idx:]

# Temporarily save to storage prior to loading into the HuggingFace ecosystem
with open(train_file, 'w', encoding='utf-8') as file:
    file.write(train_set)

with open(validate_file, 'w', encoding='utf-8') as file:
    file.write(validate_set)

# Load train/validate sets into the HuggingFace ecosystem
custom_text = load_dataset('text', data_files={'train': train_file,
                                                'validate': validate_file})
print(custom_text)

# Perform clean up
remove(train_file)
remove(validate_file)

############# DETERMINE MAX SENTENCE LENGTHS FROM CORPUS #############
def max_sentence_length(file_path):
    """
    Reads a text file and returns the maximum length of characters in any sentence.

    IMPORTANT N.B.: ONLY USE THIS IN ACCORDANCE WITH THE ALLOWABLE 'MAX_LENGTH' OF YOUR CHOSEN
    PRE-TRAINED TOKENIZER. OTHERWISE DURING THE `LLM_FINETUNE` MODULE, THE INTERPRETER WILL RAISE
    'OUT OF RANGE' ERRORS. FOR INSTANCE, GPT2Tokenizer WILL ONLY ALLOW MAX_LENGTH=1024. SO, ONLY USE
    THIS FUNCTION IF THE TOKENIZER DOESN'T HAVE A SET MAX_LENGTH. OTHERWISE, IGNORE THIS STEP AND ONLY
    RUN THE TOKENIZER WHILE HARD CODING THE MAX_LENGTH ARG.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into sentences using regular expressions
    sentences = re.split(r'[.!?]+', text)

    max_length = 0
    for sentence in sentences:
        # Remove leading/trailing whitespace and count characters
        sentence_length = len(sentence.strip())
        if sentence_length > max_length:
            max_length = sentence_length
    
    print(f'THE MAXIMUM NUMBER OF CHARACTERS IN YOUR CORPUS: {max_length:,}')

    return max_length

# max_length = max_sentence_length(Path(content_path) / 'content_cleaned.txt')

############# DEFINE AND RUN TOKENIZER #############
def encode_batches(batch, max_length=max_length):
    
    tokenized_data = tokenizer(batch['text'], 
                               padding='max_length', 
                               truncation=True,
                               return_attention_mask=True,
                               max_length=max_length,
                               return_tensors='pt')
    return tokenized_data

tokenized_data = custom_text.map(encode_batches, batched=True, batch_size=batch_size)
tokenized_data.save_to_disk(Path(LLM_pretrained_path))