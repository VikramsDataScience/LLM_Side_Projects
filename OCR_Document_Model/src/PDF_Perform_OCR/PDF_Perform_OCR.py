import pytesseract
from PIL import Image

# Function to perform OCR using Tesseract on an image
def perform_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text