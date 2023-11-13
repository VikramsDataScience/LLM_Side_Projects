from os import path, remove
from torch import cuda
from tqdm.auto import tqdm
import json
from pathlib import Path
from datasets import load_dataset # ! pip install datasets
from bs4 import BeautifulSoup

# Verify if GPU is being used
print('Is GPU available?: ', cuda.is_available())

clean_list = []

# Declare paths for the raw and cleaned JSON files
raw_data_path = 'C:/Sample Data/'
content_path = 'C:/Sample Data/content_cleaned.json'

with open(Path(raw_data_path) / 'ads-50k.json', 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

# Declare a function to strip HTML tags and perform some other Pre-processing
def stripTags(input_string):

    soup = BeautifulSoup(input_string, 'html.parser')
    string = input_string.replace('.', '. ')

    return soup.get_text(strip=True), string

# Strip HTML tags from the 'content' object and append it to a new JSON dictionary 
# Since the json.dump() call in the loop is an 'append' statement, if the file exists delete it. Otherwise, the json.dump() call will append the dictionary without limit (i.e. objects will duplicate)
if path.exists(Path(raw_data_path) / 'content_cleaned.json'):
    remove(Path(raw_data_path) / 'content_cleaned.json')

for i in tqdm(range(0, len(raw_data)), desc='Data Pre-Processing Progress'):

    content_cleaned = {
        'content': stripTags(raw_data[i]['content'])
    }

    # Append to a JSON list
    clean_list.append(content_cleaned)

    # Dump JSON into a file
    with open(Path(raw_data_path) / 'content_cleaned.json', 'a') as f:
        json.dump(content_cleaned, f)