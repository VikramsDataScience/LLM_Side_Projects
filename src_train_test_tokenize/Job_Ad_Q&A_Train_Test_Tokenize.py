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

# Verify if GPU is being used
print('Is GPU available?: ', torch.cuda.is_available())

# Declare paths for the raw and cleaned JSON files
raw_data_path = 'C:/Sample Data/'
content_path = 'C:/Sample Data/content_cleaned.json'
# Declare path for saved models
saved_models_path = 'C:/Users/Vikram Pande/venv/saved_models'

# First, the path for the files needs to be declared in a dictionary before performing the train/test split (the typical Python Path() doesn't seem to work with HuggingFace)
train_test_dict = {'train': content_path, 
                   'test': content_path}

clean_content = load_dataset('json', 
                    data_files=train_test_dict, 
                    split='train')
clean_content = clean_content.train_test_split(test_size=0.3, shuffle=True)

print(clean_content['train'])
print('TRAINING SET SAMPLE: ', clean_content['train'][-1].values())
print('TEST SET SAMPLE: ', clean_content['test'][-1])

# Create a checkpoint (i.e. pretrained data), and initialise the tokens
# IMPORTANT: Before selecting a sentence embedding pretrained_model, please review the updated performance metrics for other commonly used models here (https://www.sbert.net/docs/pretrained_models.html#model-overview)
pretrained_model = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

# Some of the following code is taken from 'sentence-transformers/all-mpnet-base-v2' model card as part of best practice implementation (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Return the token embeddings and attention_mask averages
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Recursively tokenize the data by mapping 
tokenized_data = clean_content.map(lambda x: tokenizer(x['content'], padding=True, truncation=True, return_tensors='pt'), 
                                   batched=True)

# Display the tokenized data to ensure that 'input_ids' and 'attention_mask' features are included
print(tokenized_data)

# Mount the Pre-trained Model and inputs to the GPU
device = torch.device("cuda")
model.to(device)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**tokenized_data['train']['input_ids'])

# Perform pooling
sentence_embeddings = mean_pooling(model_output, tokenized_data['train']['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print("SENTENCE EMBEDDINGS: ", sentence_embeddings)

