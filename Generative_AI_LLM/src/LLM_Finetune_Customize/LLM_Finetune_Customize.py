from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, TextDataset, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_from_disk
import torch
from pathlib import Path
import logging
import yaml

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

LLM_pretrained_path = global_vars['LLM_pretrained_path']
training_log_path = global_vars['training_log_path']
model_output_path = global_vars['model_output_path']
pretrained_model = global_vars['pretrained_model']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pretrained_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)
custom_pre_trained_docs = load_from_disk(Path(LLM_pretrained_path) / 'LLM_Trained_Tokenized')

train_dataset = TextDataset(custom_pre_trained_docs)
# When using a Pretrained model (such as https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) that contain special tokens like [INST], [/INST], etc. set 'mlm': Masked Language Modelling to 'True'
data_collator = DataCollatorForLanguageModeling(tokenizer=pretrained_tokenizer, mlm=False)

# Define training arguments before running the Trainer
training_args = TrainingArguments(per_device_train_batch_size=2, # Number of tokens per batch 
                                  num_train_epochs=3,
                                  logging_dir=training_log_path,
                                  output_dir=model_output_path)

# Run the Trainer to enable Transfer Learning (i.e. fine tune the custom dataset)
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset)
trainer.train()

# model.save_pretrained(Path(LLM_pretrained_path) / 'fine_tuned_LLM')