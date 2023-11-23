from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset, Dataset # ! pip install datasets
from pathlib import Path

# Declare paths for the raw and cleaned JSON files
raw_data_path = 'C:/Sample Data/'
content_path = 'C:/Sample Data/content_cleaned.json'
# Declare path for saved models
saved_models_path = Path('C:/Users/Vikram Pande/Job_Ad_QA_HuggingFace/saved_models')

# IMPORTANT: Before selecting a sentence embedding pretrained_model, please review the updated performance metrics for other commonly used models here (https://www.sbert.net/docs/pretrained_models.html#model-overview)
pretrained_model = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'

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
clean_content_dataset = Dataset.from_dict({'content': clean_content['content'][0:100]}) # [0:10000]

# Verify if GPU is being used and mount the Pre-trained Model and inputs to the GPU
print('Is GPU available?: ', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# CLS Pooling - Take output from the first token
def cls_pooling(last_hidden_state):
    return last_hidden_state[:, 0]

def encode_batch(batch):
    # Tokenize a batch of sentences
    encoded_input = tokenizer(batch['content'], padding='max_length', truncation=True, return_tensors='pt').to(device)

    # Disable SGD to eliminate randomizing error loss and improve inference
    with torch.no_grad():
        # Pass tensors to the model
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output.last_hidden_state)

    return {'content': embeddings}

# Write sentence embedding
query = 'Write a job ad for a software engineer'

# Encode query and pass it as a list within a dictionary
query_emb = encode_batch({'content': [query]})

# Batch size for processing the documents
batch_size = 100

# Use map function to process the documents in batches and save the model for later downstream
docs = clean_content_dataset.map(encode_batch, batched=True, batch_size=batch_size)
tokenizer.save_pretrained(saved_models_path) # , docs

# Extract embeddings from the list of dictionaries
doc_embeddings = [item['content'] for item in docs]

# Convert the list of embeddings to a tensor
doc_embeddings = torch.cat([torch.tensor(embedding).unsqueeze(0) for embedding in doc_embeddings], dim=0)

# Compute dot score between the query and all document embeddings
scores = torch.mm(query_emb['content'], doc_embeddings.transpose(0, 1).to(device))[0]

# Combine corpus & scores
doc_score_pairs = list(zip(clean_content_dataset['content'], scores.cpu().tolist()))

# Sort by decreasing score and capture top 5 ranked results
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)[:5]

# Output predictions & scores
for doc, score in doc_score_pairs:
    print(f'ACCURACY SCORE: {score}')
    print(f'JOB AD: {doc}')