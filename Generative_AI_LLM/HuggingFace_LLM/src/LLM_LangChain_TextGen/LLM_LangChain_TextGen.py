from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer
from pathlib import Path
import logging
import yaml

logger = logging.getLogger('LLM_Finetune')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Error_Logs/LLM_Finetune_log.log'))
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
pretrained_model = global_vars['pretrained_HG_model']

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model_name = f'C:/Sample Data/Job_Ad_QA_data/saved_models/llm_saved_models/fine_tuned_LLM_{model_ver}'
tokenizer.save_pretrained(model_name)

hf = HuggingFacePipeline.from_model_id(model_id=model_name,
                                       task='text-generation',
                                       pipeline_kwargs={'max_new_tokens': 200},
                                       model_kwargs={'temperature': 0.6,
                                                     'top_p': 0.9,
                                                     'do_sample': True,
                                                     'no_repeat_ngram_size': 2},
                                       device_map='auto') # Can replace with 'device=0' to explicitly specify GPU

# Create template and build chain
template = """
        QUESTION: {question}
        ANSWER: """
prompt = PromptTemplate.from_template(template)
chain = prompt | hf

question = 'Write a job ad for a Senior Data Scientist'
print(chain.invoke({'question': question}))