import fitz # PyMuPDF for PDF and image data extraction
from torchvision import transforms
import PIL
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
    try:
        image = Image.open(image_path).convert('RGB')
    except PIL.UnidentifiedImageError:
        pass
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image)
        return input_tensor.unsqueeze(0)

def extract_images_from_pdf(files_path, extracted_images_path):
    pdf_document = fitz.open(files_path)
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        images_list = page.get_images(full=True)

        for img_index, img_info in enumerate(images_list):
            img_id = img_info[0]
            base_image = pdf_document.extract_image(img_id)
            image_bytes = base_image['image']
            image = Image.open(io.BytesIO(image_bytes))
            # Save to storage location (each image will represent one page)
            pdf_filename = path.splitext(path.basename(files_path))[0] # Extract filename without .pdf extension
            image.save(open(Path(extracted_images_path) / f'docid_{pdf_filename}_page_{page_number + 1}_img_{img_index}.png', 'w'))

# Iterate and load PDF documents, read all pages in document & extract images
for pdf_file in tqdm(listdir(files_path), desc='PDF Image Extraction Progress'):
    # If the output folder doesn't exist, create folder
    if not path.exists(extracted_images_path):
        makedirs(extracted_images_path)
    
    try:
        id = start_id
        pdf_images = extract_images_from_pdf(f'{files_path}/0{id:.4f}.pdf', extracted_images_path)
    except fitz.fitz.FileNotFoundError: # Handle any FileNotFoundErrors caused by research papers being removed from arXiv server
        pass
    else:
        id += 0.0001    