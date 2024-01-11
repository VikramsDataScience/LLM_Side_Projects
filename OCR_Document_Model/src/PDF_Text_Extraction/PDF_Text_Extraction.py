from os import path, remove
import json
import fitz # PyMuPDF for PDF and image data extraction
from pathlib import Path
import yaml
import logging
from tqdm.auto import tqdm

logger = logging.getLogger('PDF_Text_Extraction')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/PDF_Text_Extraction_Error_Log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects/OCR_Document_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

# Load global variables from config YAML file and declare local variables
files_path = global_vars['files_path']
start_id = global_vars['start_id']
end_id = global_vars['end_id']
extracted_text_path = global_vars['extracted_text_path']
id = start_id

# Function to recursively open the PDFs with PyMuPDF, extract text and save to storage
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
        
        # Declare JSON dictionary to store extracted text
        extracted_text = {
            'research_text': text
        }

        with open(Path(extracted_text_path) / 'extracted_text.json', 'a') as f:
            json.dump(extracted_text, f)

    return text

# Since the json.dump() call in the loop is an 'append' statement, if the file exists delete it. Otherwise, the json.dump() call will append the dictionary without limit (i.e. objects will duplicate)
if path.exists(Path(extracted_text_path) / 'extracted_text.json'):
    remove(Path(extracted_text_path) / 'extracted_text.json')

# Iterate through the list of downloaded PDFs
with tqdm(total=int((end_id - start_id) / 0.0001) + 1, desc='Text Extraction Progress') as pbar:
    while id <= end_id:
        extract_text_from_pdf(f'{files_path}/0{id:.4f}.pdf')
        id += 0.0001
        pbar.update(1)