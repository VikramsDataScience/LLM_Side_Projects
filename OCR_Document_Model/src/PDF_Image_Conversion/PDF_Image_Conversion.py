from os import listdir
from os.path import join, splitext, basename
import fitz # PyMuPDF for PDF and image data extraction
from PIL import Image
from pathlib import Path
import yaml
import logging
from tqdm.auto import tqdm

logger = logging.getLogger('PDF_Image_Conversion')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/PDF_Image_Conversion_Log.log'))
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
extracted_PNG_path = global_vars['extracted_PNG_path']
id = start_id

# Create a PNG images folder if it doesn't exist
extracted_PNG_path = Path(extracted_PNG_path)
extracted_PNG_path.mkdir(parents=True, exist_ok=True)

# Function to recursively open the PDFs with PyMuPDF, extract text and save to storage
def convert_pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        # Perform conversion of scanned page to image
        pixmap = page.get_pixmap()
        image = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)

        # Save image with original filename and page number
        image_filename = f'{splitext(basename(pdf_path))[0]}_page_{page_num + 1}.png'
        image_path = join(output_folder, image_filename)
        image.save(image_path)
    
    doc.close()

# Iterate through the list of downloaded PDFs
for pdf_file in tqdm(listdir(files_path), desc='PDF Image Conversion Progress'):
    try:
        convert_pdf_to_images(f'{files_path}/0{id:.4f}.pdf', extracted_PNG_path)
    except fitz.fitz.FileNotFoundError: # Handle any FileNotFoundErrors caused by research papers being removed from arXiv server
        pass

    id += 0.0001