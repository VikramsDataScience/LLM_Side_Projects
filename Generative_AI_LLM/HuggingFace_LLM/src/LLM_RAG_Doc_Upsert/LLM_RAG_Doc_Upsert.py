from pinecone import Pinecone, ServerlessSpec
from os import getenv
from dotenv import load_dotenv
from pathlib import Path
import logging
import yaml

logger = logging.getLogger('LLM_RAG_Evaluation')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Error_Logs/LLM_RAG_Evaluation_log.log'))
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

API_Key_path = global_vars['API_Key_file']
content_path = global_vars['content_path']
batch_size = global_vars['batch_size']
index_name = 'job-ad-index'

# Load API Key from ENV file, initialise connection with Pinecone and create an index
load_dotenv(dotenv_path=API_Key_path)
api_key = getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key, pool_threads=30) # Set 'pool_threads' to limit server requests during the batch upsert phase

pc.create_index(name=index_name,
                dimension=128,
                metric='dotproduct',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                ))

index = pc.Index(index_name)

with open(Path(content_path) / 'content_cleaned.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()
