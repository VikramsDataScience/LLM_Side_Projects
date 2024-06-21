import json
import re
import ftfy
from ftfy import TextFixerConfig
from tqdm.auto import tqdm
from pathlib import Path
import yaml
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger('Q&A_PreProcessing')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Error_Logs/Q&A_PreProcessing_log.log'))
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
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')
    # logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

# Declare paths for the raw and cleaned JSON files
clean_list = []
content_path = global_vars['content_path']

with open(Path(content_path) / 'ads-50k.json', 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

# Declare a function to identify and strip HTML & Unicode tags from the corpus
def stripTags(input_string):

    string = BeautifulSoup(input_string, 'html.parser')
    string = string.get_text(strip=True)
    string = string.replace('\n', ' ').replace('\\', ' ')
    # Heuristic method to fix common issues with unicode
    string = ftfy.fix_text(string, TextFixerConfig(restore_byte_a0=True, fix_line_breaks=True, decode_inconsistent_utf8=True))
    string = string.encode('utf-8').decode('unicode_escape')
    string = string.encode('utf-8').decode('ISO-8859-1')
    string = string.encode('latin-1').decode('utf-8')
    string = string.replace('Â', '').replace('Â·', '').replace('â¢', '')

    return string

# Recursively strip HTML tags and store in a TXT file
for i in tqdm(range(0, len(raw_data)), desc='Text Pre-Processing Progress'):
    
    content_cleaned = stripTags(raw_data[i]['content'])
    clean_list.append(content_cleaned)

# Store the 'clean_list' list into a TXT file, and encode with 'utf-8' to comply with OpenAI's tiktoken encoder/decoder
with open(Path(content_path) / 'content_cleaned.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(clean_list))