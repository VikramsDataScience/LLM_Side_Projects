import requests
import os
from urllib.parse import urlparse
from pathlib import Path
import yaml

# IMPORTANT N.B.: arXiv do have an official API. However, the response from the API is a form of XML called Atom. Since, I'm currently interested in developing OCR for PDF 
# documents, I've elected for the manual implementation represented by the following component module by downloading the research papers directly from 
# specific URLs based on their arXiv ID numbers (i.e. the 'start_id' and 'end_id' variables). However, it should be noted that Engineering best practice would dictate using the API

# Load the file paths and global variables from YAML config file
config_path = Path('C:/Users/Vikram Pande/Side_Projects/OCR_Document_QA')

with open(config_path / 'config.yml', 'r') as file:
    global_vars = yaml.safe_load(file)

# Declare global variables from config YAML file
files_path = global_vars['files_path']
start_id = global_vars['start_id']
end_id = global_vars['end_id']

def download_pdf_from_url(url, file_path='.'):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Extract filename from the Content-Disposition header, if available
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            filename = content_disposition.split('filename=')[-1].strip('";')
        else:
            # If the header is not present, parse the URL to get the filename
            parsed_url = urlparse(url)
            filename = parsed_url.path.split('/')[-1]
        
        # Combine file_path and filename
        save_path = os.path.join(file_path, filename)

        # Save the PDF content to a file
        with open(save_path, 'wb') as pdf_file:
            pdf_file.write(response.content)    

        print(f'PDF downloaded successfully. Saved as: {filename}')

    except requests.exceptions.RequestException as e:
        print(f'Error downloading PDF: {e}')

# Call function and recursively download the research papers as PDFs
while start_id <= end_id:
    url = f'https://arxiv.org/pdf/0{start_id:.4f}.pdf'
    download_pdf_from_url(url, files_path)
    start_id += 0.0001