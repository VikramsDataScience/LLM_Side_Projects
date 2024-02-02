import cv2 # Run 'pip install opencv-python scikit-image' to install OpenCV and scikit-image
from PIL import Image
import imutils
from os import listdir
import numpy as np
from skimage import transform
from pathlib import Path
import yaml
import logging
from tqdm.auto import tqdm

logger = logging.getLogger('Image_Preprocessing')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects/Error_Logs/Image_Preprocessing_Log.log'))
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
id = global_vars['start_id']
extracted_images_path = global_vars['extracted_images_path']
preprocessed_images_path = global_vars['preprocessed_images_path']
preprocessed_image_width = global_vars['preprocessed_image_width']
preprocessed_image_height = global_vars['preprocessed_image_height']
scale_x = global_vars['scale_x']
scale_y = global_vars['scale_y']

# Create a preprocessed images folder if it doesn't exist
preprocessed_images_path = Path(preprocessed_images_path)
preprocessed_images_path.mkdir(parents=True, exist_ok=True)

############# DEFINE IMAGE PREPROCESSING CLASS WITH ASSOCIATED METHODS #############
class ImagePreprocessing:
    @staticmethod # Use '@staticmethod' decorator to remove any dependencies on the instance state (i.e. remove the requirement for a 'self' parameter in the class)
    def load_image(image_path):
        """
        Load an image from the specified file path.

        Parameters:
        - image_path (str): The file path to the image.

        Returns:
        - numpy.ndarray: A NumPy array representing the loaded image.

        Raises:
        - FileNotFoundError: If the specified file path does not exist.
        - Exception: If there is an error during the image loading process.

        Note:
        The function uses the OpenCV library to read the image with the flag cv2.IMREAD_UNCHANGED,
        which loads the image as is, including the alpha channel for transparency if present.
        """
        return Image.open(image_path)

    @staticmethod
    def rescale_image(image, new_width, new_height, fx, fy):
        """
        Rescales the input image to the specified width and height.

        Parameters:
        - fx: To maintain the aspect ratio and preservation of the display quality scale up/down the x-axis
        - fy: To maintain the aspect ratio and preservation of the display quality scale up/down the y-axis
        - image (numpy.ndarray): Input image.
        - new_width (int): Width of the rescaled image.
        - new_height (int): Height of the rescaled image.

        Returns:
        numpy.ndarray: Rescaled image.
        """
        image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a numpy array.")
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if len(image.shape) != 3:
            raise ValueError("Input 'image' must be a 3-dimensional array (height, width, channels).")
        
        # Use 'INTER_LINEAR' as a general purpose interpolation technique for all resized images
        return cv2.resize(image, (new_width, new_height), fx, fy, interpolation=cv2.INTER_LINEAR)

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
        # If the image is already in grayscale or has a single channel
        if len(image.shape) == 2 or image.shape[2] == 1:
            return image
        # Convert 3-channel image to grayscale
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert 4-channel BGRA to grayscale ignoring the alpha channel
        elif image.shape[2] == 4:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        
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
    def apply_threshold(image):
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
        binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return binary_image
    
    @staticmethod
    def normalize_image(image):
        """
        Apply a distance transform (such as Euclidean L2 distance) which will calculate the distance to the nearest zero pixel 
        for each pixel in the image. Perform normalization of pixel values of the input image to the range [0, 1].
        Convert the distance transform back to unsigned 8-bit integer.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        numpy.ndarray: Normalized image with pixel values in the range [0, 1].
        """
        image = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        image = cv2.normalize(image, image, 0, 1.0, cv2.NORM_MINMAX)
        image = (image * 255).astype('uint8')

        # Perform and return the threshold distance transform using Otsu's method
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

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
        # If image is not 8-bit single-channel, recast the data type to apply adaptive threshold
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def localize_text(image):
        """
        Localizes text in the input image using morphological operations.

        Parameters:
        - image (numpy.ndarray): Input image.
        - kernel_size (tuple): Size of the structuring element kernel. Default is (5, 5).

        Returns:
        numpy.ndarray: Image with localized text.
        """
        # The getStructuringElement() operation denoises elements and disconnects connected blobs on the open operation (i.e. cv2.MORPH_OPEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def image_contours(image, extracted_image):
        """
        Extract contours in a binary image by isolating the foreground blobs. Find all the contours (characters and noise)
        and perform a filter to only keep those that are 35px wide and 100px tall.
        """
        contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        chars = []

        for c in contours:
            # Compute bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # Perform check to see if contour is at least 35px wide & 100px tall. If so, consider it a digit
            if w >= 35 and h >= 100:
                chars.append(c)

        if len(chars) > 0:
            # Compute the convex hull of the characters    
            chars = np.vstack([chars[i] for i in range(0, len(chars))])
            hull = cv2.convexHull(chars)

            # Allocate memory for the convex hull mask, draw the convex hull as a mask and enlarge via dilation
            extracted_image = np.array(extracted_image)
            mask = np.zeros(extracted_image.shape[:2], dtype='uint8')
            cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.dilate(mask, None, iterations=2)

            # To resolve any assertion errors convert to unsigned 8-bit integer
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            result = cv2.bitwise_and(image, image, mask=mask)     
            return result
        else:
            return image

    @staticmethod
    def rotate_image(image, angle=15):
        """
        Rotate the given image by the specified angle. This is useful to bring text into horizontal alignment for the OCR engine.
        If the text isn't horizontally aligned, the OCR engine may not be able to identify the text accurately.

        Parameters:
        - image: The input image.
        - angle: The angle of rotation in degrees (default is 15 degrees).

        Returns:
        The rotated image.
        """
        return transform.rotate(image, angle=angle)

############# INTANTIATE THE CLASS AND CALL IMAGE PREPROCESSING METHODS IN SYNCHRONOUS PIPELINE ORDER #############
ImgPreprocess = ImagePreprocessing()

def preprocess_and_save(image_path, output_folder):
    image = ImgPreprocess.load_image(image_path)
    file_name = Path(image_path).name

    # Apply preprocessing steps in order
    preprocessed_image = ImgPreprocess.rescale_image(image, fx=scale_x, fy=scale_y, new_width=preprocessed_image_width, new_height=preprocessed_image_height)
    preprocessed_image = ImgPreprocess.reduce_noise(preprocessed_image)
    preprocessed_image = ImgPreprocess.convert_to_grayscale(preprocessed_image)
    preprocessed_image = ImgPreprocess.edge_detection(preprocessed_image)
    preprocessed_image = ImgPreprocess.apply_threshold(preprocessed_image)
    preprocessed_image = ImgPreprocess.normalize_image(preprocessed_image)
    preprocessed_image = ImgPreprocess.deskew_image(preprocessed_image)
    preprocessed_image = ImgPreprocess.apply_adaptive_threshold(preprocessed_image)
    preprocessed_image = ImgPreprocess.localize_text(preprocessed_image)
    preprocessed_image = ImgPreprocess.image_contours(preprocessed_image, extracted_image=image)
    # preprocessed_image = ImgPreprocess.rotate_image(preprocessed_image)

    # Save preprocessed images
    output_folder = Path(output_folder) / file_name
    # Convert back to image from numpy array and to grayscale 'convert('L')' for Tesseract. Or 'convert('RGB')' to convert array to RGB
    pil_image = Image.fromarray(preprocessed_image)
    pil_image = pil_image.convert('L')
    try:
        pil_image.save(output_folder)
    except Exception as e:
        print(f'ERROR SAVING IMAGE: {e}')

############# LOOP THROUGH IMAGE PREPROCESSING STEPS FOR THE EXTRACTED IMAGES #############
for page_filename in tqdm(listdir(extracted_images_path), desc='Image Preprocessing Progress'):
    page_path = Path(extracted_images_path) / page_filename
    # Extract page number from the filename
    page_number = page_filename.split('_')[-1].split('.')[0]
    preprocess_and_save(page_path, preprocessed_images_path)