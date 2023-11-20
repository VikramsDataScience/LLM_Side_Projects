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
from datasets import load_dataset, Dataset # ! pip install datasets
from pathlib import Path

# Declare paths for the raw and cleaned JSON files
raw_data_path = 'C:/Sample Data/'
content_path = 'C:/Sample Data/content_cleaned.json'
# Declare path for saved models
saved_models_path = 'C:/Users/Vikram Pande/venv/saved_models'

# IMPORTANT: Before selecting a sentence embedding pretrained_model, please review the updated performance metrics for other commonly used models here (https://www.sbert.net/docs/pretrained_models.html#model-overview)
pretrained_model = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'

# Create a checkpoint (i.e. pretrained data), and initialize the tokens
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

# Load the dataset
train_test_dict = {'train': content_path, 'test': content_path}
clean_content = load_dataset('json', data_files=train_test_dict, split='train')

# Convert 'clean_content' list to a Dataset
clean_content_dataset = Dataset.from_dict({'content': clean_content['content'][0:100]})

# Verify if GPU is being used and mount the Pre-trained Model and inputs to the GPU
print('Is GPU available?: ', torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# CLS Pooling - Take output from the first token
def cls_pooling(last_hidden_state):
    return last_hidden_state[:, 0]

def encode_batch(batch):
    # Tokenize a batch of sentences
    encoded_input = tokenizer(batch['content'], padding=True, truncation=True, return_tensors='pt').to(device)

    # Disable SGD to eliminate randomizing error loss and improve inference
    with torch.no_grad():
        # Pass tensors to the model
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output.last_hidden_state)

    return {'content': embeddings}

# Write sentence embedding
query = "Write a job ad for a software engineer"

# Encode query
query_emb = encode_batch({'content': [query]})  # Pass the query as a list within a dictionary

# Batch size for processing the documents
batch_size = 8

# Use map function to process the documents in batches
docs = clean_content_dataset.map(encode_batch, batched=True, batch_size=batch_size)

# Extract embeddings from the list of dictionaries
doc_embeddings = [item['content'] for item in docs]

# Convert the list of embeddings to a tensor
doc_embeddings = torch.cat([embedding.unsqueeze(0) for embedding in doc_embeddings], dim=0)

# Compute dot score between the query and all document embeddings
scores = torch.mm(query_emb['content'], doc_embeddings.transpose(0, 1))[0]

# Ensure scores is a tensor
scores = torch.tensor(scores)

# Combine corpus & scores
doc_score_pairs = list(zip(clean_content_dataset['content'], scores.cpu().tolist()))

# Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

# Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)