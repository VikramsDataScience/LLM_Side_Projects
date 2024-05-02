from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from os import getenv
from dotenv import load_dotenv
from pathlib import Path
import logging
import yaml

logger = logging.getLogger('LLM_RAG_Evaluation')
logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler()
error_handler = logging.FileHandler(Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Error_Logs/LLM_RAG_Evaluation_log.log'))
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/Generative_AI_LLM/HuggingFace_LLM')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    logger.error(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

API_Key_path = global_vars['API_Key_file']
content_path = global_vars['content_path']
batch_size = global_vars['batch_size']
index_name = 'job-ad-index'

# Load API Key from ENV file, initialise connection with Pinecone and create an index
load_dotenv(dotenv_path=API_Key_path)
pineconce_api_key = getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pineconce_api_key)

if index_name not in pc.list_indexes().names():
    print(f'{index_name} not found. Creating index...')
    pc.create_index(name=index_name,
                    dimension=128,
                    metric='dotproduct',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    ))

index = pc.Index(index_name)

def text_chunking(file_path, chunk_size=1000, chunk_overlap=0):
    """
    Perform chunking on a sample of text from a TXT file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text_file = file.read()

    # Split corpus into much smaller sample
    split_ratio = 0.02
    split_idx = int(split_ratio * len(text_file))
    sample_set = text_file[:split_idx]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(sample_set)

    # Verify doc chunks
    print(chunks[:2])

    return chunks

# Create Vector Store
def create_vectors(index_name):
    """
    For a given 'index_name', create vectorised embeddings.
    """
    embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = pinecone.Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    print(f'Embeddings for Index: \'{index_name}\' successfully upserted!')

    return vectors

text_chunks = text_chunking(file_path=Path(content_path) / 'content_cleaned.txt')
# vectors = create_vectors(index_name)