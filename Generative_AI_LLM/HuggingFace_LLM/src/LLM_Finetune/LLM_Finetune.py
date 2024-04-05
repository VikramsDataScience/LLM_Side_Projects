
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import Dataset
import evaluate
import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
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
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Error_Logs/LLM_TrainTokenize_log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Generative_AI_LLM/HuggingFace_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

model_ver = global_vars['model_ver']
LLM_pretrained_path = global_vars['LLM_pretrained_path']
training_log_path = global_vars['training_log_path']
model_output_path = global_vars['model_output_path']
pretrained_model = global_vars['pretrained_HG_model']
seed = global_vars['seed']

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

############# LOAD PRETRAINED TOKENIZER/MODEL, CUSTOM DATA, AND DEFINE HYPERPARAMETERS #############
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)
metric = evaluate.load('accuracy')

# Define Hyperparameters and load tokenized Train/Validate sets left by upstream 'LLM_Tokenize' module
train_set = Dataset.load_from_disk(Path(LLM_pretrained_path)/ 'train').shuffle(seed=seed)
validate_set = Dataset.load_from_disk(Path(LLM_pretrained_path) / 'validate').shuffle(seed=seed)
print('TRAIN SET:\n', train_set, '\nVALIDATE SET:\n', validate_set)

# Perform train/validate on about half of the data to save on training time
train_set_sample = train_set.select(range(45000))
validate_set_sample = validate_set.select(range(5000))

# Call compute on metric to calculate the accuracy of the predictions. Before passing the predictions to compute, the logits will need to be converted to predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

############# LOAD PRETRAINED TOKENIZER/MODEL, CUSTOM DATA, AND DEFINE HYPERPARAMETERS #############
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)

# Define Hyperparameters and load tokenized Train/Validate sets left by upstream 'LLM_Tokenize' module
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

train_set = Dataset.load_from_disk(Path(LLM_pretrained_path)/ 'train')
validate_set = Dataset.load_from_disk(Path(LLM_pretrained_path) / 'validate')
print('TRAIN SET:\n', train_set, 'VALIDATE SET:\n', validate_set)

############# DEFINE AND RUN TRAINING LOOP #############
# Collate pretrained tokenizer into batches for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                mlm=False, 
                                                return_tensors='pt')

# Define training arguments to be used by the Trainer
training_args = TrainingArguments(per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  learning_rate=1e-4,
                                  weight_decay=0.01,
                                  evaluation_strategy='steps',
                                  num_train_epochs=3,
                                  eval_steps=1,
                                  warmup_steps=3, # Should be about 20% of the 'num_train_epochs'
                                  logging_dir=training_log_path,
                                  output_dir=model_output_path,
                                  auto_find_batch_size=True,
                                  gradient_accumulation_steps=2, # Occurs prior to the Optimizer, and affects the effective batch size and the frequency of optimization steps (can use between 2 to 32)
                                #   fp16=True, # Use Mixed Precision training (roughly equivalent to setting pytorch's 'autocast()' class in the training loop) to improve training loop time
                                  seed=seed) 
                                  num_train_epochs=1.5,
                                  logging_dir=training_log_path,
                                  output_dir=model_output_path,
                                  auto_find_batch_size=True,
                                  gradient_accumulation_steps=64, # Occurs prior to the Optimizer, and affects the effective batch size and the frequency of optimization steps (can use between 2 to 32)
                                  fp16=True) # Use Mixed Precision training (roughly equivalent to setting pytorch's 'autocast()' class in the training loop) to improve training loop time

# Run the Trainer to enable Transfer Learning and fine tune based on the custom dataset
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_set_sample,
                  eval_dataset=validate_set_sample,
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)

optimizer = trainer.create_optimizer()
num_training_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
num_eval_steps = len(trainer.get_eval_dataloader()) * training_args.eval_steps

# Define Learning Rate Scheduler with Linear Warmup
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=training_args.warmup_steps,
                                            num_training_steps=num_training_steps)

# Define a custom Learning Rate Scheduler callback
class LRSchedulerCallback(TrainerCallback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_step_end(self, args, state, control, 
                    model=None, 
                    tokenizer=None, 
                    optimizer=optimizer, 
                    lr_scheduler=scheduler, 
                    train_dataloader=num_training_steps,
                    eval_dataloader=num_eval_steps):
        self.scheduler.step()

lr_scheduler_callback = LRSchedulerCallback(scheduler)
trainer.add_callback(lr_scheduler_callback)
                  train_dataset=train_set,
                  eval_dataset=validate_set,
                  tokenizer=tokenizer,
                  optimizers=(optimizer, scheduler.step()))
trainer.train()

# Save finetuned model for downstream module)
model.save_pretrained(Path(LLM_pretrained_path) / f'fine_tuned_LLM_{model_ver}')
# print(f'Training loop completed! Model parameters saved in the following storage location:\nPath(LLM_pretrained_path) / \'fine_tuned_LLM_{model_ver}\'')
