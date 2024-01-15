import pytesseract
import fitz # PyMuPDF for PDF and image data extraction
from torchvision import transforms
from PIL import Image
from pathlib import Path
from os import listdir, path, makedirs
import io
import yaml
import logging
from tqdm.auto import tqdm

logger = logging.getLogger('PDF_Image_Extraction')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/PDF_Image_Extraction_Error_Log.log'))
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
files_path = global_vars['files_path']
start_id = global_vars['start_id']
extracted_images_path = global_vars['extracted_images_path']
id = start_id

# Function to preprocess image using PyTorch transforms
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image)
    return input_tensor.unsqueeze(0)

def extract_images_from_pdf(output_path):
    # If the output folder doesn't exist, create folder
    if not path.exists(output_path):
        makedirs(output_path)
    
    # Load PDF document, read all pages in document & extract images
    for pdf_file in tqdm(listdir(f'{files_path}/0{id:.4f}.pdf'), desc='PDF Image Extraction Progress'):
        try:
            pdf_document = fitz.open(pdf_file)
        except fitz.fitz.FileNotFoundError: # Handle any FileNotFoundErrors caused by research papers being removed from arXiv server
            pass
        else:
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]
                images = page.get_images(full=True)

                for img_index, img_info in enumerate(images):
                    img_index += 1
                    img_index = img_info[0]
                    base_image = pdf_document.extract_image(img_index)
                    image_bytes = base_image['image']
                    image = Image.open(io.BytesIO(image_bytes))

                    # Perform image Preprocessing
                    image_preprocess = preprocess_image(f'{files_path}/0{id:.4f}.pdf')

                    # Save to storage location
                    save_path = path.join(output_path, f'page_{page_number + 1}_img_{img_index}.png')
                    image_preprocess.save(save_path)

                pdf_document.close()
        
        id += 0.0001

extract_images_from_pdf(extracted_images_path)

# Function to perform OCR using Tesseract on an image
def perform_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text