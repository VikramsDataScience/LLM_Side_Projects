from os import path, remove
from torch import cuda
from tqdm.auto import tqdm
import json
from pathlib import Path
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
if path.exists(Path(raw_data_path) / 'content_cleaned.json'):
    remove(Path(raw_data_path) / 'content_cleaned.json')

# Recursively strip HTML tags and display a progress bar in the for loop
for i in tqdm(range(0, len(raw_data)), desc='Data Pre-Processing Progress'):
    content_cleaned = {
        'content': stripTags(raw_data[i]['content'])
    }

    # Append to a JSON list
    clean_list.append(content_cleaned)

    # Dump JSON into a file
    with open(Path(raw_data_path) / 'content_cleaned.json', 'a') as f:
        json.dump(content_cleaned, f)