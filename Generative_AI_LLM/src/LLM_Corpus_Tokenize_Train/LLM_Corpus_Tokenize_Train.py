from transformers import GPT2LMHeadModel
import tiktoken
from sklearn.model_selection import train_test_split
import torch
from lightning.pytorch.tuner import Tuner
from lightning import LightningModule
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader
import numpy as np
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
pretrained_HG_model = global_vars['pretrained_HG_model']
LLM_pretrained_path = global_vars['LLM_pretrained_path']
batch_size = global_vars['batch_size']
model_ver = global_vars['model_ver']
learning_rate = global_vars['learning_rate']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pretrained model from HuggingFace Hub and mount to the GPU
pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_HG_model).to(device)

def tokenize_data(content_file_path):
    """ 
    Load custom text corpus from the TXT file created by the 'LLM_PreProcessing' module.
    IMPORTANT N.B.: If using OpenAI's 'tiktoken', the TXT file must be 'utf-8' encoded to ensure lossless execution. 
    Otherwise tiktoken will generate lossy (data will leak) tokens during the encode/decode steps. Consequently, it 
    could lead to losing important text information.
    """
    with open(Path(content_file_path), 'r', encoding='utf-8') as f:
        custom_text = f.read()

    # Use OpenAI's GPT-4 Byte Pair Encoding (BPE) tokenizer to encode the Job ad text corpus
    tokenizer = tiktoken.get_encoding('cl100k_base')
    # 'allow_special' arg refers to correctly tokenizing any special characters that may exist. It can be set to 'all', 'none', or a custom list of special tokens to allow/disallow
    print('Tokenizing (Encoding) custom TXT file...')
    encoded_text = tokenizer.encode(custom_text, allowed_special='all')
    print('Encoding successfully completed!')
    
    return encoded_text

def train_test(tokenzized_training_set):
    """
    Split into Train & Test sets, convert to Tensor data structure and save the
    'test_data' set to storage location for validation further downstream.
    """
    train_data, test_data = train_test_split(tokenzized_training_set, 
                                            test_size=0.10, 
                                            random_state=314)
    print('Saving test set as:', f'LLM_test_data_{model_ver}.npy')

    np.save(Path(LLM_pretrained_path) / f'LLM_test_data_{model_ver}.npy', test_data)
    print('Testing set saved successfully! \nTesting set storage location:', Path(LLM_pretrained_path) / f'LLM_test_data_{model_ver}.npy')
    del test_data

    return train_data

encoded_text = tokenize_data(content_file)
train_data = train_test(encoded_text)

train_tensor = torch.LongTensor(train_data).to(device)

print(f'No. of parameters in the Pretrained \'{pretrained_HG_model}\' model: {pretrained_model.num_parameters():,}')

class FineTuneLRFinder(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = Model(...)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

model = FineTuneLRFinder()
trainer = Trainer(...)
tuner = Tuner(trainer)
# This will find the learning rate automatically and sets hparams.learning_rate to that learning rate
tuner.lr_find(model)

# Divide training set into batches, pin_memory allows for faster transfer of data to the GPU, and num_workers enables asynchronous data fetching that won't block GPU computation 
train_batches = DataLoader(train_tensor, 
                           batch_size=batch_size, 
                           shuffle=True, 
                           pin_memory=True, 
                           num_workers=4)

# Define and commence training loop
pretrained_model.train()
