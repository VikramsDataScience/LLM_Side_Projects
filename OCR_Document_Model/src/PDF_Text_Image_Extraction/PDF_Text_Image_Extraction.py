import pytesseract
import torch
import fitz # PyMuPDF for PDF and image data extraction
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import yaml
import logging

logger = logging.getLogger('PDF_Text_Image_Extraction')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/PDF_Text_Image_Extraction_Error_Log.log'))
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

# If GPU is available, instantiate a device variable to use the GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Declare global variables from config YAML file
files_path = global_vars['files_path']
start_id = global_vars['start_id']
end_id = global_vars['end_id']

def pdf_text_extract(path):
    doc = fitz.open(path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
        print(text)
    return text

# Call function and recursively open the PDFs with PyMuPDF
pdf_text_extract(f'{files_path}/0{start_id:.4f}.pdf')