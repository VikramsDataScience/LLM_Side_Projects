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

############# DEFINE IMAGE PREPROCESSING CLASS WITH ASSOCIATED METHODS #############
class ImagePreprocessing:
    @staticmethod # Use '@staticmethod' decorator to remove any dependencies on the instance state (i.e. remove the requirement for a 'self' parameter in the class)
    def load_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def rescale_image(image, new_width, new_height):
        """
        Rescales the input image to the specified width and height.

        Parameters:
        - image (numpy.ndarray): Input image.
        - new_width (int): Width of the rescaled image.
        - new_height (int): Height of the rescaled image.

        Returns:
        numpy.ndarray: Rescaled image.
        """
        return cv2.resize(image, (new_width, new_height))

    @staticmethod
    def reduce_noise(image):
        """
        In order to separate the noise in the images, apply blurring. Sharper images lose their detail, so using a low-pass filter
        has the effect of reducing the amount of noise and detail in the image.
        
        Applies Gaussian blur to the input image for noise reduction.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        numpy.ndarray: Image with reduced noise.
        """
        return cv2.GaussianBlur(image, (5,5), 0)

    @staticmethod
    def convert_to_grayscale(image):
        """
        Transformations within RGB space like adding/removing the alpha channel, reversing the channel order, 
        conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as conversion to/from grayscale.

        Parameters:
        - image (numpy.ndarray): Input color image.

        Returns:
        numpy.ndarray: Grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def edge_detection(image, lower_threshold=50, upper_threshold=150):
        """
        Edge detection works better when the image is first converted to grayscale and smoothed or blurred 
        (as has been done in the convert_to_grayscale() & reduce_noice() functions).

        Performs edge detection on the input image using the Canny edge detector.

        Parameters:
        - image (numpy.ndarray): Input image.
        - lower_threshold (int): Lower threshold for edge detection. Default is 50.
        - upper_threshold (int): Upper threshold for edge detection. Default is 150.

        Returns:
        numpy.ndarray: Image with detected edges.
        """
        return cv2.Canny(image, lower_threshold, upper_threshold)

    @staticmethod
    def apply_threshold(image, threshold_value=128):
        """
        Applies a fixed-level threshold to each array element.

        The function applies fixed-level thresholding to a multiple-channel array. The function is typically used to 
        get a bi-level (binary) image out of a grayscale image ( compare could be also used for this purpose) or for 
        removing a noise, that is, filtering out pixels with too small or too large values. There are several types of thresholding supported by the function. 
        They are determined by type parameter.

        Also, the special values THRESH_OTSU or THRESH_TRIANGLE may be combined with one of the above values. 
        In these cases, the function determines the optimal threshold value using the Otsu's or Triangle algorithm and uses it instead of the specified thresh.

        Note: Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.
        Parameters:
        src	input array (multiple-channel, 8-bit or 32-bit floating point).
        dst	output array of the same size and type and the same number of channels as src.
        - thresh:	threshold value.
        - maxval:	maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
        - type:	thresholding type (see ThresholdTypes).
        Returns:
        The computed threshold value if Otsu's or Triangle methods are used.
        """
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    @staticmethod
    def enhance_contrast(image):
        """
        Enhances the contrast of the input image using histogram equalization.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        numpy.ndarray: Image with enhanced contrast.
        """
        return exposure.equalize_hist(image)

    @staticmethod
    def deskew_image(image):
        """
        Deskews the input image by rotating it to align with the predominant text orientation.

        Parameters:
        - image (numpy.ndarray): Input image containing text.

        Returns:
        numpy.ndarray: Deskewed image.
        """
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

    @staticmethod
    def apply_adaptive_threshold(image):
        """
        Applies adaptive thresholding to the input image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        numpy.ndarray: Thresholded image.
        """
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def localize_text(image, kernel_size=(5,5)):
        """
        Localizes text in the input image using morphological operations.

        Parameters:
        - image (numpy.ndarray): Input image.
        - kernel_size (tuple): Size of the structuring element kernel. Default is (5, 5).

        Returns:
        numpy.ndarray: Image with localized text.
        """
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def resample_image(image, new_width, new_height):
        """
        Resamples the input image to the specified dimensions using interpolation.

        Parameters:
        - image (numpy.ndarray): Input image.
        - new_width (int): Width of the resampled image.
        - new_height (int): Height of the resampled image.

        Returns:
        numpy.ndarray: Resampled image.
        """
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image):
        """
        Normalizes pixel values of the input image to the range [0, 1].

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        numpy.ndarray: Normalized image with pixel values in the range [0, 1].
        """
        return image.astype(np.float32) / 255.0

    @staticmethod
    def rotate_image(image, angle=15):
        """
        Rotate the given image by the specified angle.

        Parameters:
        - image: The input image.
        - angle: The angle of rotation in degrees (default is 15 degrees).

        Returns:
        The rotated image.
        """
        return transform.rotate(image, angle=angle)

############# INTANTIATE CLASS AND CALL IMAGE PREPROCESSING STEPS IN THEIR RESPECTIVE ORDER #############
ImgPreprocess = ImagePreprocessing()

def preprocess_and_save(image_path, output_folder):
    image = ImgPreprocess.load_image(image_path)

    # Apply preprocessing steps in order
    preprocessed_image = ImgPreprocess.rescale_image(image, new_width=preprocessed_image_width, new_height=preprocessed_image_height)
    preprocessed_image = ImgPreprocess.reduce_noise(preprocessed_image)
    preprocessed_image = ImgPreprocess.convert_to_grayscale(preprocessed_image)
    preprocessed_image = ImgPreprocess.edge_detection(preprocessed_image)
    preprocessed_image = ImgPreprocess.apply_threshold(preprocessed_image)
    preprocessed_image = ImgPreprocess.enhance_contrast(preprocessed_image)
    preprocessed_image = ImgPreprocess.deskew_image(preprocessed_image)
    preprocessed_image = ImgPreprocess.apply_adaptive_threshold(preprocessed_image)
    preprocessed_image = ImgPreprocess.localize_text(preprocessed_image)
    preprocessed_image = ImgPreprocess.resample_image(preprocessed_image, new_width=preprocessed_image_width, new_height=preprocessed_image_height)
    preprocessed_image = ImgPreprocess.normalize_image(preprocessed_image)
    preprocessed_image = ImgPreprocess.rotate_image(preprocessed_image)

    # Save preprocessed images
    output_path = path.join(Path(output_folder), '.png')
    cv2.imwrite(preprocessed_image, output_path)

############# LOOP THROUGH IMAGE PREPROCESSING STEPS FOR THE EXTRACTED IMAGES #############
for image in tqdm(listdir(extracted_images_path), desc='Image Preprocessing Progress'):
    # If the output folder doesn't exist, create the folder
    if not path.exists(preprocessed_images_path):
        makedirs(preprocessed_images_path)

    preprocess_and_save(image, preprocessed_images_path)
    