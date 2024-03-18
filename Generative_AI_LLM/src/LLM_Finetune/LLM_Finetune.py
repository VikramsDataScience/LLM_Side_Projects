from transformers import GPT2LMHeadModel, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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
pretrained_model = global_vars['pretrained_HG_model']
content_file_path = global_vars['content_file']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

############# LOAD PRETRAINED TOKENIZER/MODEL AND DEFINE HYPERPARAMETERS #############
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)

# Define Hyperparameters
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

print('Loading upstream Tokenized Batches from disk location...')
tokenized_batches = Dataset.load_from_disk(Path(LLM_pretrained_path)/ 'train')

############# DEFINE AND RUN TRAINING PIPELINE #############
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

# Define training arguments to be used by the Trainer
training_args = TrainingArguments(per_device_train_batch_size=8,
                                  num_train_epochs=1,
                                  logging_dir=training_log_path,
                                  output_dir=model_output_path,
                                  auto_find_batch_size=True,
                                  gradient_accumulation_steps=64, # Occurs prior to the Optimizer, and affects the effective batch size and the frequency of optimization steps (can use between 2 to 32)
                                  fp16=True) # Use Mixed Precision training (roughly equivalent to setting pytorch's 'autocast()' class in the training loop) to improve training loop time

# Run the Trainer to enable Transfer Learning and fine tune based on the custom dataset
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=tokenized_batches,
                  tokenizer=tokenizer,
                  optimizers=(optimizer, scheduler.step()))
trainer.train()

# Save finetuned model for downstream module
# model.save_pretrained(Path(LLM_pretrained_path) / 'fine_tuned_LLM')