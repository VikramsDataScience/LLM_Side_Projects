from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset, Dataset
import yaml
import logging
from pathlib import Path

logger = logging.getLogger('Q&A_TrainTokenize')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/Q&A_TrainTokenize_log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects/Job_Ad_QA_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

# Declare paths for the raw and cleaned JSON files
content_path = global_vars['content_path'] / 'content_cleaned.json'
saved_models_path = global_vars['saved_models_path']
# IMPORTANT: Before selecting a sentence embedding pretrained_model, please review the updated performance metrics for other commonly used models here (https://www.sbert.net/docs/pretrained_models.html#model-overview)
pretrained_model = global_vars['pretrained_model']

# Create a checkpoint (i.e. pretrained data), and initialize the tokens
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

# Load the dataset
train_test_dict = {'train': content_path, 
                   'test': content_path}

clean_content = load_dataset('json', 
                             data_files=train_test_dict, 
                             split='train')

# Convert 'clean_content' list to a Dataset
clean_content_dataset = Dataset.from_dict({'content': clean_content['content']}) # [0:10000]

# Verify if GPU is being used and mount the Pre-trained Model and inputs to the GPU
print('Is GPU available?: ', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# CLS Pooling - Take output from the first token
def cls_pooling(last_hidden_state):
    return last_hidden_state[:, 0]

def encode_batch(batch):
    # Tokenize a batch of sentences and mount to the GPU
    encoded_input = tokenizer(batch['content'], padding='max_length', truncation=True, return_tensors='pt').to(device)

    # Disable SGD to eliminate randomizing error loss and improve inference
    with torch.no_grad():
        # Pass tensors to the model
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output.last_hidden_state)

    return {'content': embeddings}

# Batch size for processing the documents
batch_size = 100

# Use map function to process the documents in batches and save the model for further downstream
docs = clean_content_dataset.map(encode_batch, batched=True, batch_size=batch_size)
docs.save_to_disk(Path(saved_models_path))