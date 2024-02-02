from os import listdir
from os.path import join
import pytesseract
from PIL import Image
from pathlib import Path
import yaml
import logging
from tqdm.auto import tqdm

logger = logging.getLogger('Perform_OCR')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/Perform_OCR_Log.log'))
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

# Load global variables from config YAML file
extracted_PNG_path = global_vars['extracted_PNG_path']
OCR_Results_path = global_vars['OCR_Results_path']
start_id = global_vars['start_id']
id = start_id

# Function to perform OCR using Tesseract on an image
def perform_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

# Iterate through the list of PNG files and perform OCR
for png_file in tqdm(listdir(extracted_PNG_path), desc='Performing OCR on Extracted PNG Images'):
    try:
        ocr_result = perform_ocr(png_file)
    except IOError: # Handle any errors that Image.open(image_path) may raise caused by research papers being removed from arXiv server
        pass
    
    results_path = join(OCR_Results_path, f'{png_file}_OCR_Result.txt')
    with open(results_path, 'w', encoding='utf-8') as results_file:
        results_file.write(ocr_result)

    # id += 0.0001