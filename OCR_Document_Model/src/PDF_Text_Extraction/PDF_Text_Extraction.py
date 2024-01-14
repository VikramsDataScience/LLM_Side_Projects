from os import listdir
from os.path import join, splitext
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
extracted_text_path = global_vars['extracted_text_path']
id = start_id

# Function to recursively open the PDFs with PyMuPDF, extract text and save to storage
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

# Iterate through the list of downloaded PDFs
for pdf_file in tqdm(listdir(files_path), desc='Text PDF Extraction Progress'):
    try:
        pdf_text = extract_text_from_pdf(f'{files_path}/0{id:.4f}.pdf')
    except fitz.fitz.FileNotFoundError: # Handle any FileNotFoundErrors
        pass
    else:
        # Save extracted text to a text file
        output_text_file = join(extracted_text_path, f'{splitext(pdf_file)[0]}_text.txt')
        with open(output_text_file, 'w', encoding='utf-8') as text_file:
            text_file.write(pdf_text)

    id += 0.0001