from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tiktoken
from sklearn.model_selection import train_test_split
import torch
from lightning.pytorch.tuner import Tuner
from lightning import LightningModule
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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
model_ver = global_vars['model_ver']
learning_rate = global_vars['learning_rate']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pretrained model from HuggingFace Hub and mount to the GPU
pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_HG_model).to(device)
pretrained_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_HG_model)

def encode_data(content_file_path):
    """ 
    Load custom text corpus from the TXT file created by the 'LLM_PreProcessing' module.
    IMPORTANT N.B.: If using OpenAI's 'tiktoken', the TXT file must be 'utf-8' encoded to ensure lossless execution. 
    Otherwise tiktoken will generate lossy (data will leak) tokens during the encode/decode steps. Consequently, it 
    could lead to losing important text information.

    N.B.: 'allow_special' arg in the encode() refers to correctly tokenizing any special characters that may exist. 
    It can be set to 'all', 'none', or a custom list of special tokens to allow/disallow
    """
    with open(Path(content_file_path), 'r', encoding='utf-8') as f:
        custom_text = f.read()

    # Use OpenAI's GPT-4 Byte Pair Encoding (BPE) tokenizer to encode the Job ad text corpus
    tokenizer = tiktoken.get_encoding('cl100k_base')
    print('Tokenizing (Encoding) custom TXT file...')
    encoded_custom_text = tokenizer.encode(custom_text, allowed_special='all')
    print(f'Encoding completed.\nThere are {len(encoded_custom_text):,} tokens in the corpus')

    # Perform check to ensure that encoding process is lossless
    try:
        assert tokenizer.decode(encoded_custom_text) == custom_text
        print('Assertion test complete. Byte Pair Encoding correctly encoded and verified!')
    except AssertionError:
        print('WARNING: Byte Pair Encoding failed. Please check your text corpus for any lossy issues during encoding process.')
    
    return encoded_custom_text

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

class FineTuneLRFinder(LightningModule):
    """
    Use Pytorch Lightening's experimental Learning Rate finder and Batch Size Finder 
    to utilise the most optimal Learning Rates and Batch sizes during Backpropagation.
    """
    def __init__(self, pretrained_model, pretrained_tokenizer, custom_data, learning_rate):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.tokenizer = pretrained_tokenizer
        self.custom_data = custom_data
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass to run through the data in the model.
        """
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
        return outputs

    def training_step(self):
        """
        Define and run training loop
        """
        

    def collate_fn(self):
        """
        To ensure that matrix multiplication is performed on rectangular matrices, run this function
        within the 'collate_fn' argument of the DataLoader() to pad the shorter sentences with 0s (or any other 
        value you like) by calling Pytorch's pad_sequence() function.
        """
        return pad_sequence(self.custom_data, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def train_dataloader(self):
        """
        Divide training set into batches, pad shorter sentences using the defined 'collate_fn()', pin_memory allows 
        for faster transfer of data to the GPU, and num_workers enables asynchronous data fetching that won't 
        block GPU computation.
        """
        return DataLoader(self.custom_data,
                        collate_fn=self.collate_fn,
                        batch_size=self.batch_size | self.hparams.batch_size, 
                        shuffle=True, 
                        pin_memory=True, 
                        num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=(self.lr | self.learning_rate))

encoded_text = encode_data(content_file)
train_data = train_test(encoded_text)
train_tensor = torch.LongTensor(train_data).to(device)

print(f'No. of parameters in the Pretrained \'{pretrained_HG_model}\' model: {pretrained_model.num_parameters():,}')

model = FineTuneLRFinder(pretrained_model, 
                         pretrained_tokenizer, 
                         train_tensor, 
                         learning_rate)

trainer = Trainer(accelerator='gpu', 
                  devices=1, 
                  inference_mode=torch.no_grad())
tuner = Tuner(trainer)

# This will scale the batches by doubling the 'batch_size' until training set maximum is reached or an Out of Memory (OOM) error is raised - whichever comes first
tuner.scale_batch_size(model, mode='power')

# Get the new suggested LR
lr_finder = tuner.lr_find(model)
new_lr = lr_finder.suggestion()

# Update the model's hparams with the new LR, train the model and save the checkpoint
model.hparams.lr = new_lr
trainer.fit(model)
trainer.save_checkpoint(Path(LLM_pretrained_path) / 'trained_model_and_weights.ckpt')