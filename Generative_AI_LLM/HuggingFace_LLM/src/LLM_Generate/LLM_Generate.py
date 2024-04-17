from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
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
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Generative_AI_LLM/HuggingFace_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

LLM_pretrained_path = global_vars['LLM_pretrained_path']
training_log_path = global_vars['training_log_path']
model_output_path = global_vars['model_output_path']
pretrained_model = global_vars['pretrained_HG_model']
model_ver = global_vars['model_ver']
seed = global_vars['seed']

# Initialise argparse to accept and parse user prompts
parser = argparse.ArgumentParser('Predictions_Scoring')
parser.add_argument('--query', type=str, help='Please enter query string/prompt to generate a model\'s response',
                    default='Write a job ad for a Senior Data Scientist')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
# Load the Finetuned model left by the upstream 'LLM_Finetune' module and mount to GPU
finetuned_model = GPT2LMHeadModel.from_pretrained(Path(LLM_pretrained_path) / f'fine_tuned_LLM_{model_ver}').to(device)

def generate_text(prompt, context='', temperature=0.6, top_k=0, top_p=0.90, max_new_tokens=200):
    """
    - 'temperature': Modify the 'temperature' arg to stipulate the stochasticity of the model's generated response by 
    adjusting the value of the applied Softmax layer from the converted logit (i.e. higher temperature will make the 
    model's response less deterministic). This arg can be used in conjunction with 'top_k' as a scaling method to 
    temper the number of top_k n-grams that are part of the sampling scheme. However, experimentation has shown that
    'top_p' can be a better choice and temperature doesn't have to be used at all.
    - 'top_k' (requires 'do_sample'=True): In Top-K sampling, the K most likely next words are filtered and the probability mass is 
    redistributed among only those K next words.
    - 'top_p' (requires 'do_sample'=True): The 'nucleus' sampling method works by sampling only from the most likely K words. Top-p sampling 
    chooses from the smallest possible set of words whose cumulative probability exceeds the probability p.
    """
    input_text = f'{context} {prompt}'
    encoding = tokenizer.encode_plus(input_text, return_tensors='pt', padding=True).to(device)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    # set_seed(seed) # Set a seed if you wish to reproduce results
    
    output = finetuned_model.generate(input_ids, 
                                        attention_mask=attention_mask, 
                                        max_new_tokens=max_new_tokens,
                                        do_sample=True,
                                    #   temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p
                                        )
    
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # Strip user query from model generated response
    generated_text = ''.join(generated_text[len(str(input_text)) + 1:])
    print('YOUR QUERY:\n', prompt)

    return generated_text

conversation_context = ''

while True:
    user_input = input('Reply to chatbot (type \'EXIT CHAT!\' to exit): ')
    if user_input.upper() == 'EXIT CHAT!':
        break
    generated_response = generate_text(prompt=user_input, context=conversation_context)
    print('MODEL GENERATED RESPONSE:\n', generated_response)

    # Update context
    conversation_context = f'{conversation_context} {user_input} {generated_response}'
