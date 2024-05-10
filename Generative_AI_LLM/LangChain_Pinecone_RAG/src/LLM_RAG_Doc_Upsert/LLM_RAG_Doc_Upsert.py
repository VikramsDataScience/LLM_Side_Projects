from pinecone import Pinecone, ServerlessSpec
import pinecone
from sentence_transformers import SentenceTransformer
import itertools
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
pinecone_api_key = getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)

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

# Text chunking
def text_chunking(file_path, chunk_size=1000, batch_size=batch_size):
    """
    Perform chunking as an iterable on a sample of text from a TXT file that is of a given 
    batch_size. Pinecone best practice indicates max batch_size=100.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text_file = file.read()

    # Split corpus into much smaller sample
    chunk_id = 0
    split_ratio = 0.0001
    split_idx = int(split_ratio * len(text_file))
    sample_set = text_file[:split_idx]
    print('TOTAL NUMBER OF CHARACTERS TO BE UPSERTED: ', len(sample_set))

    for i in range(0, len(sample_set), chunk_size):
        chunk = sample_set[i: i + chunk_size]
        # Yield tuple of 'chunk_id' and 'chunk' of text
        yield (f'chunk-{chunk_id}', chunk)
        chunk_id += 1

# Helper function that defines an iterable
def chunks(iterable, batch_size=batch_size):
    """
    Helper function to break an iterable into 'batch_size' chunks.
    This was derived from Pinecone's documentation 
    (https://docs.pinecone.io/guides/data/upsert-data#upsert-records-in-batches)
    """
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def create_vectors(index_name):
    """
    For a given 'index_name', create vectorised embeddings.
    """
    embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = pinecone.Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    print(f'Embeddings for Index: \'{index_name}\' successfully upserted!')

    return vectors

# Generate chunk_id/chunk pairs
chunked_data = text_chunking(file_path=Path(content_path) / 'content_cleaned.txt')

for batch in chunks(chunked_data, batch_size=batch_size):
    index.upsert(vectors=[batch])
    print(f'Length of batch {len(batch)} upserted to Pinecone index name: \'{index_name}\'')
        # Once Upserting is complete, print to verify index's namespaces, dimensions, vector_count, etc.
        # print(f'Embeddings for Index: \'{index_name}\' successfully upserted!\n', 
        #         f'STATS FOR \'{index_name}\':\n', index.describe_index_stats())