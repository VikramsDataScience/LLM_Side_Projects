from transformers import GPT2LMHeadModel
import argparse
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
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

# Initialise argparse
parser = argparse.ArgumentParser('LLM_Corpus_Tokenize_Train')
parser.add_argument('--query', type=str, help='Please enter query string or prompt to generate a model\'s response',
                    default='Write a job ad for a Machine Learning Engineer')
args = parser.parse_args()

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects/Generative_AI_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

content_file = global_vars['content_file']
pretrained_HG_model = global_vars['pretrained_HG_model']
LLM_pretrained_path = global_vars['LLM_pretrained_path']
model_ver = global_vars['model_ver']
learning_rate = global_vars['learning_rate']
batch_size = global_vars['batch_size']
num_epochs = global_vars['num_epochs']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pretrained model from HuggingFace Hub and mount to the GPU
pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_HG_model).to(device)
print(f'No. of parameters in the Pretrained \'{pretrained_HG_model}\' model: {pretrained_model.num_parameters():,}')

# Load the pre-processed 'custom_corpus' left by the 'LLM_Corpus_PreProcessing' upstream module
with open(Path(content_file), 'r', encoding='utf-8') as f:
    custom_corpus = f.readlines()

# Apply text Feature Extraction to the 'custom_corpus'
vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(custom_corpus)

# Use OpenAI's GPT-4 Byte Pair Encoding (BPE) tokenizer to encode the Job ad text corpus
tokenizer = tiktoken.get_encoding('cl100k_base')

def retrieve_relevant_docs(input_text, top_k=3):
    """
    Apply feature extraction techniques such as TFIDF to extract valuable information from
    the 'custom_corpus'.
    """
    input_vector = vectorizer.transform([input_text])
    similarities = corpus_vectors.dot(input_vector.T).toarray().ravel()
    top_doc_indices = similarities.argsort()[-top_k:][::-1]
    top_docs = [custom_corpus[i] for i in top_doc_indices]
    return top_docs

class ContextDataSet(Dataset):
    """
    This class inherits the 'torch.utils.data.Dataset' class from pytorch to map key to data samples
    (inputs and targets sequences generated using tiktoken).
    """
    def __init__(self, corpus, inputs):
        """ 
        Load custom text corpus from the TXT file created by the 'LLM_PreProcessing' module.
        IMPORTANT N.B.: If using OpenAI's 'tiktoken', the TXT file must be 'utf-8' encoded to ensure lossless execution. 
        Otherwise tiktoken will generate lossy (data will leak) tokens during the encode/decode steps. Consequently, it 
        could lead to losing important text information.

        N.B.: 'allow_special' arg in the encode() refers to correctly tokenizing any special characters that may exist. 
        It can be set to 'all', 'none', or a custom list of special tokens to allow/disallow
        """
        self.inputs = inputs
        self.corpus = corpus
        self.targets = []
        print('Tokenizing (Encoding) custom corpus...')
        for input_text in inputs:
            relevant_docs = retrieve_relevant_docs(input_text)
            target_text = ' '.join(relevant_docs)
            target_ids = tokenizer.encode(target_text, allowed_special='all')
            
            # Perform check to ensure that encoding process is lossless
            try:
                assert tokenizer.decode(target_ids) == target_text
            except AssertionError:
                print('WARNING: Byte Pair Encoding contains inconsistencies. Please check your text corpus for any lossy issues during encoding process.')

            self.targets.append(torch.tensor(target_ids))
        if AssertionError != True:
            print(f'Encoding and Assertion test completed. Byte Pair Encoding (BPE) correctly encoded and verified!\nThere are {len(target_ids):,} tokens in the \'target_ids\'')

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        input_ids = tokenizer.encode(input_text)
        target_ids = self.targets[idx]
        return torch.tensor(input_ids), target_ids

def generate_response(user_input, max_length=128, batch_size=32):
    """
    Model generated response with attention masks based on user input query.
    """
    input_ids = [tokenizer.encode(text) for text in user_input]
    attention_masks = [[1] * len(ids) for ids in input_ids]

    input_tensors = [torch.tensor(ids).to(device) for ids in input_ids]
    attention_tensors = [torch.tensor(masks).to(device) for masks in attention_masks]

    input_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=tokenizer.eot_token)
    attention_tensors = pad_sequence(attention_tensors, batch_first=True, padding_value=0)

    input_tensors = input_tensors.unsqueeze(0)
    attention_tensors = attention_tensors.unsqueeze(0)
    
    dataset = TensorDataset(input_tensors, attention_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    responses = []
    for input_tensor, attention_tensor in dataloader:
        output_ids = pretrained_model.generate(input_tensor, 
                                  max_length=max_length,
                                  attention_masks=attention_tensor)

        batch_responses = [tokenizer.decode(output_ids[i].tolist()) for i in range(len(output_ids))]
        responses.extend(batch_responses)

    return response

################# Run class to extract input/target QA pairs and generate response to user query #################
context_data = ContextDataSet(corpus=custom_corpus, inputs=args.query)
response = generate_response(user_input=args.query)
