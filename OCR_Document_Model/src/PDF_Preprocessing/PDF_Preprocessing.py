import cv2 # Run 'pip install opencv-python scikit-image' to install OpenCV and scikit-image
from os import path, listdir, makedirs
import numpy as np
from skimage import io, color, exposure, morphology, transform
from pathlib import Path
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
extracted_images_path = global_vars['extracted_images_path']
preprocessed_images_path = global_vars['preprocessed_images_path']
preprocessed_image_width = global_vars['preprocessed_image_width']
preprocessed_image_height = global_vars['preprocessed_image_height']

############# DEFINE IMAGE PREPROCESSING FUNCTIONS AND SAVE TO STORAGE #############
def load_image(image_path):
    return cv2.imread(image_path)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def rescale_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def apply_threshold(image, threshold_value=128):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def enhance_contrast(image):
    return exposure.equalize_hist(image)

def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    centre = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def apply_adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def localize_text(image, kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def resample_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def rotate_image(image, angle=15):
    return transform.rotate(image, angle=angle)

def preprocess_and_save(image_path, output_folder):
    image = load_image(image_path)

    # Apply preprocessing steps
    preprocessed_image = convert_to_grayscale(image)
    preprocessed_image = reduce_noise(preprocessed_image)
    preprocessed_image = apply_threshold(preprocessed_image)
    preprocessed_image = enhance_contrast(preprocessed_image)
    preprocessed_image = deskew_image(preprocessed_image)
    preprocessed_image = apply_adaptive_threshold(preprocessed_image)
    preprocessed_image = localize_text(preprocessed_image)
    preprocessed_image = resample_image(preprocessed_image, new_width, new_height)
    preprocessed_image = normalize_image(preprocessed_image)

    # Save preprocessed images
    output_path = path.join(Path(output_folder), '.png')
    save_image(preprocessed_image, output_path)

############# APPLY IMAGE PREPROCESSING FOR THE EXTRACTED IMAGES #############
for image in tqdm(listdir(extracted_images_path), desc='Image Preprocessing Progress'):
    # If the output folder doesn't exist, create the folder
    if not path.exists(extracted_images_path):
        makedirs(extracted_images_path)

    preprocess_and_save(image, preprocessed_images_path)
    