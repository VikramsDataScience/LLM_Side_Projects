from os import path, remove
from torch import cuda
from tqdm.auto import tqdm
import json
from pathlib import Path
import yaml
import logging
from bs4 import BeautifulSoup

# Verify if GPU is being used
print('Is GPU available?: ', cuda.is_available())

clean_list = []

logger = logging.getLogger('Q&A_PreProcessing')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/Q&A_PreProcessing_log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects/Job_Ad_QA_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

# Declare paths for the raw and cleaned JSON files
content_path = global_vars['content_path']

with open(Path(content_path) / 'ads-50k.json', 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

# Declare a function to strip HTML tags and perform some other Pre-processing
def stripTags(input_string):

    string = input_string.encode("ascii", "ignore")
    string = string.decode()
    string = BeautifulSoup(input_string, 'html.parser')
    string = string.get_text(strip=True)
    string = (string.replace('\u2022', '').replace('\u00a0', ' ').replace('\u2019', '').replace('\u00b7', '')
              .replace('\u2013', '').replace('\u2026', '').replace('\u200b', '').replace('\u2018', '')
              .replace('\u201c', '').replace('\u201d', '').replace('\u00e9', 'e').replace('\uff1a', '')
              .replace('\u00bd', '').replace('\u0101', '').replace('\u0113', '').replace('\u016b', 'ū')
              .replace('\u014d', 'ō').replace('\n', ' ').replace('\u00ae', '').replace('\u2122', '')
              .replace('\u00ad', '').replace('\\', ''))

    return string

# Strip HTML tags from the 'content' object and append it to a new JSON dictionary 
# Since the json.dump() call in the loop is an 'append' statement, if the file exists delete it. Otherwise, the json.dump() call will append the dictionary without limit (i.e. objects will duplicate)
if path.exists(Path(content_path) / 'content_cleaned.json'):
    remove(Path(content_path) / 'content_cleaned.json')

# Recursively strip HTML tags and display a progress bar in the for loop
for i in tqdm(range(0, len(raw_data)), desc='Data Pre-Processing Progress'):
    content_cleaned = {
        'content': stripTags(raw_data[i]['content'])
    }

    # Append to a JSON list
    clean_list.append(content_cleaned)

    # Dump JSON into a file
    with open(Path(content_path) / 'content_cleaned.json', 'a') as f:
        json.dump(content_cleaned, f)