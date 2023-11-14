import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, get_scheduler
from transformers import BertConfig, BertModel, BertTokenizer, DataCollatorWithPadding, TrainingArguments
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
import evaluate # ! pip install evaluate
from datasets import load_dataset # ! pip install datasets
from pathlib import Path

# Verify if GPU is being used
print('Is GPU available?: ', torch.cuda.is_available())

# Declare paths for the raw and cleaned JSON files
raw_data_path = 'C:/Sample Data/'
content_path = 'C:/Sample Data/content_cleaned.json'
# Declare path for saved models
saved_models_path = 'C:/Users/Vikram Pande/venv/saved_models'

# IMPORTANT: Before selecting a sentence embedding pretrained_model, please review the updated performance metrics for other commonly used models here (https://www.sbert.net/docs/pretrained_models.html#model-overview)
pretrained_model = 'sentence-transformers/all-mpnet-base-v2'

# Create a checkpoint (i.e. pretrained data), and initialise the tokens
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

# First, the path for the files needs to be declared in a dictionary before performing the train/test split (the typical Python Path() doesn't seem to work with HuggingFace)
train_test_dict = {'train': content_path, 
                   'test': content_path}

clean_content = load_dataset('json', 
                    data_files=train_test_dict, 
                    split='train')
#clean_content = clean_content.train_test_split(test_size=0.3, shuffle=True)

# print(clean_content['train'])
# print('TRAINING SET SAMPLE: ', clean_content['train'][-1].values())
# print('TEST SET SAMPLE: ', clean_content['test'][-1])

# Mount the Pre-trained Model and inputs to the GPU
device = torch.device("cuda")
model.to(device)

# Some of the following code is taken from 'sentence-transformers/all-mpnet-base-v2' model card as part of best practice implementation (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Return the token embeddings and attention_mask averages
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(text):
    # Recursively tokenize the data by mapping 
    tokenized_data = clean_content.map(tokenizer(text['content'], 
                                                 padding=True, 
                                                 truncation=True, 
                                                 return_tensors='pt'), 
                                    batched=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return mean_pooling(model_output=model_output, attention_mask=tokenized_data['attention_mask'])

generate_embeddings(clean_content)
# Ensure that 'input_ids' and 'attention_mask' included in the tokenizations
# print(tokenized_data)

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**tokenized_data['train'])

# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, tokenized_data['train']['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
# print("SENTENCE EMBEDDINGS: ", sentence_embeddings)

# Dump pretrained configurations to a specified path
#tokenized_data.save_pretrained(Path(saved_models_path))