from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import Dataset
import torch
from pathlib import Path
import yaml

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Generative_AI_LLM/HuggingFace_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

model_ver = global_vars['model_ver']
LLM_pretrained_path = global_vars['LLM_pretrained_path']
training_log_path = global_vars['training_log_path']
model_output_path = global_vars['model_output_path']
pretrained_model = global_vars['pretrained_HG_model']
seed = global_vars['seed']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

############# LOAD PRETRAINED TOKENIZER/MODEL, CUSTOM DATA, AND DEFINE HYPERPARAMETERS #############
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model).to(device)

# Define Hyperparameters and load tokenized Train/Validate sets left by upstream 'LLM_Tokenize' module
train_set = Dataset.load_from_disk(Path(LLM_pretrained_path)/ 'train').shuffle(seed=seed)
validate_set = Dataset.load_from_disk(Path(LLM_pretrained_path) / 'validate').shuffle(seed=seed)
print('TRAIN SET:\n', train_set, '\nVALIDATE SET:\n', validate_set)

# Perform train/validate on a small sample of the corpus to save on training time
train_set_sample = train_set.select(range(500))
validate_set_sample = validate_set.select(range(50))

############# DEFINE AND RUN TRAINING LOOP #############
# Collate pretrained tokenizer into batches for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                mlm=False, 
                                                return_tensors='pt')

# Define training arguments to be used by the Trainer
training_args = TrainingArguments(per_device_train_batch_size=8, # Set default batch size for training, but should be overwritten by the 'auto_find_batch_size' arg
                                  per_device_eval_batch_size=8, # Set default batch size for evaluation, but should be overwritten by the 'auto_find_batch_size' arg
                                  learning_rate=1e-4, # Set initial low LR for updating weights, but the LRSchedulerCallback() will take over with Linear Warmup and Cosine Decay
                                  weight_decay=0.01, # Set L2 Regularization to prevent overfitting
                                  evaluation_strategy='steps', # Perform evaluation at end of each step/epoch
                                  num_train_epochs=3, # Set to 3 full passes through the entire training set
                                  eval_steps=1, # Frequency of evaluation steps set to 1 (i.e. run evaluation at every step)
                                  warmup_steps=1, # Should be about 20% (arg expects 'int', so you'll need to round up) of the 'num_train_epochs'
                                  logging_dir=training_log_path,
                                  output_dir=model_output_path, # Set path for checkpointing
                                  auto_find_batch_size=True, # Automatically determine batch size that will fit on the device's memory to prevent Out of Memory errors
                                  gradient_accumulation_steps=2, # Occurs prior to the Optimizer, and affects the effective batch size and the frequency of optimization steps (can use between 2 to 32)
                                  fp16=True, # Use Mixed Precision training (roughly equivalent to setting pytorch's 'autocast()' class in the training loop) to improve training loop time
                                  seed=seed)

# Run the Trainer to enable Transfer Learning and fine tune based on the custom dataset
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_set_sample,
                  eval_dataset=validate_set_sample,
                  tokenizer=tokenizer)

optimizer = trainer.create_optimizer()
num_training_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
num_eval_steps = len(trainer.get_eval_dataloader()) * training_args.eval_steps

# Define Learning Rate Scheduler with Linear Warmup and Cosine Decay (for Linear Warmup and Linear Decay please use the 'get_linear_schedule_with_warmup()' method)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
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
trainer.train(resume_from_checkpoint=None)

# Save finetuned model for downstream module
model.save_pretrained(Path(LLM_pretrained_path) / f'fine_tuned_LLM_{model_ver}')
print(f'Training loop completed! Model parameters saved in the following storage location:\n{Path(LLM_pretrained_path)} / \'fine_tuned_LLM_{model_ver}\'')