from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import torch
from datasets import load_dataset, Dataset # ! pip install datasets
from pathlib import Path

# IMPORTANT: Before selecting a sentence embedding pretrained_model, please review the updated performance metrics for other commonly used models here (https://www.sbert.net/docs/pretrained_models.html#model-overview)
pretrained_model = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
# Paths for preprocessed corpus and saved models
content_path = 'C:/Sample Data/content_cleaned.json'
saved_models_path = Path('C:/Users/Vikram Pande/Job_Ad_QA_HuggingFace/saved_models')

# Create a checkpoint (i.e. pretrained data), and initialize the tokens
pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)

# Load the dataset
train_test_dict = {'train': content_path, 
                   'test': content_path}

clean_content = load_dataset('json', 
                             data_files=train_test_dict, 
                             split='train')

# Convert 'clean_content' list to a Dataset
clean_content_dataset = Dataset.from_dict({'content': clean_content['content'][0:200]}) # [0:10000]

# Verify if GPU is being used and mount the Pre-trained Model and inputs to the GPU
print('Is GPU available?: ', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load the tokenized docs that were generated from the 'Job_Ad_Q&A_Train_Tokenize' component
docs_tokenizer = Dataset.load_from_disk(Path(saved_models_path / 'encoded_docs.pkl'))

# Write sentence embedding
query = 'Write a job ad for a software engineer'

# Extract embeddings from the list of dictionaries
doc_embeddings = [item['content'] for item in docs_tokenizer]

# Convert to Tensor
docs_tokenizer = torch.tensor(docs_tokenizer['content']).squeeze(0)

# Convert the list of embeddings to a tensor
doc_embeddings = torch.cat([torch.tensor(embedding).unsqueeze(0) for embedding in doc_embeddings], dim=0)

# Compute dot score between the query and all document embeddings
scores = torch.mm(docs_tokenizer.to(device), doc_embeddings.transpose(0, 1).to(device))[0]

# Combine corpus & scores
doc_score_pairs = list(zip(clean_content_dataset['content'], scores.cpu().tolist()))

# Sort by decreasing score and capture top 5 ranked results
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)[:5]

# Output predictions & scores
for doc, score in doc_score_pairs:
    print(f'ACCURACY SCORE: {score}')
    print(f'JOB AD: {doc}')