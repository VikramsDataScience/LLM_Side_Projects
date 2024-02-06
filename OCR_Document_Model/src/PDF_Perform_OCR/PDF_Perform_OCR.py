from os import listdir
from os.path import join, splitext, basename
import pytesseract
from PIL import Image
from pathlib import Path
import yaml
import logging
from tqdm.auto import tqdm

# The tesseract.exe will only become available once the installation file downloaded from https://github.com/UB-Mannheim/tesseract/wiki is run after running the pip install
# After installation of the above is complete, find the exe file path and invoke Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

# Create the OCR Results folder if it doesn't exist
OCR_Results_path = Path(OCR_Results_path)
OCR_Results_path.mkdir(parents=True, exist_ok=True)

# Function to perform OCR using Tesseract on an image
def perform_ocr(image_path):
    text = pytesseract.image_to_string(image_path, lang='eng')
    return text

# Iterate through the list of PNG files and perform OCR
for png_file in tqdm(listdir(extracted_PNG_path), desc='Performing OCR on Extracted PNG Images'):
    ocr_results = []
    try:
        image_path = Path(extracted_PNG_path) / png_file
        image = Image.open(image_path)
        ocr_result = perform_ocr(image)
        ocr_results.append(ocr_result)
    except IOError: # Handle any errors that Image.open(image_path) may raise caused by research papers being removed from arXiv server
        pass
    
    file_name = f'{splitext(basename(png_file))[0]}.txt'
    results_path = join(OCR_Results_path, file_name)

    with open(results_path, 'w', encoding='utf-8') as results_file:
        results_file.write('\n'.join(ocr_results))