import pytesseract
import torch
import fitz # PyMuPDF for PDF and image data extraction
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import yaml

# Load the file paths and global variables from YAML config file
config_path = Path('C:/Users/Vikram Pande/Side_Projects/OCR_Document_QA')

with open(config_path / 'config.yml', 'r') as file:
    global_vars = yaml.safe_load(file)

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