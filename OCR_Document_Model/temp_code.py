import os
import pytesseract
import fitz
from PIL import Image
import torch
from torchvision import transforms

# Set the path to the Tesseract executable (replace with your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the path to the folder containing PDF documents
pdf_folder = 'path/to/pdf/documents'

# Set the path to the folder where you want to save the extracted text
output_folder = 'path/to/output/folder'

# Function to extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

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

# Function to perform OCR using Tesseract on an image
def perform_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each PDF file in the specified folder
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)

        # Extract text from PDF using PyMuPDF
        pdf_text = extract_text_from_pdf(pdf_path)

        # Save extracted text to a text file
        output_text_file = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_text.txt")
        with open(output_text_file, 'w', encoding='utf-8') as text_file:
            text_file.write(pdf_text)

        # Extract images from the PDF (assuming images are present)
        images_folder = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_images")
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            images.extend(page.get_pixmap().get_images())

        # Save images as separate files
        for i, img in enumerate(images):
            image_path = os.path.join(images_folder, f"image_{i + 1}.png")
            img.writePNG(image_path)

            # Perform OCR on each image using Tesseract
            image_text = perform_ocr(image_path)

            # Save OCR results to a text file
            output_ocr_file = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_image_{i + 1}_ocr.txt")
            with open(output_ocr_file, 'w', encoding='utf-8') as ocr_file:
                ocr_file.write(image_text)

print("Text extraction and OCR completed.")
