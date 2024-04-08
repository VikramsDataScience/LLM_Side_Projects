from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from pathlib import Path
import logging
import yaml
import argparse

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
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Generative_AI_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

LLM_pretrained_path = global_vars['LLM_pretrained_path']
training_log_path = global_vars['training_log_path']
model_output_path = global_vars['model_output_path']
pretrained_model = global_vars['pretrained_HG_model']

# Initialise argparse to accept and parse user prompts
parser = argparse.ArgumentParser('Predictions_Scoring')
parser.add_argument('--query', type=str, help='Please enter query string/prompt to generate a model\'s response',
                    default='Write a job ad for a Senior Data Scientist')
args = parser.parse_args()

print('Is GPU available?:', torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(pretrained_model)
# Load the Finetuned model left by the upstream 'LLM_Finetune' module and mount to GPU
finetuned_model = GPT2LMHeadModel.from_pretrained(Path(LLM_pretrained_path) / 'fine_tuned_LLM')

def generate_text(prompt, max_length=50, temperature=0.7):
    """
    Modify the 'temperature' arg to stipulate the stochasticity of the model's generated response (i.e.
    higher temperature will make the model's response less deterministic).
    """
    encoding = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.inference_mode():
        output = finetuned_model.generate(input_ids, 
                                          attention_mask=attention_mask, 
                                          max_length=max_length,
                                          max_new_tokens=100,
                                          do_sample=True,
                                          temperature=temperature)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

generated_response = generate_text(prompt=args.query)
print('MODEL GENERATED RESPONSE:\n', generated_response)