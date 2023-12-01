import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # Use this if using N-Grams for feature extraction
from pathlib import Path

# Define file paths
files_path = Path('C:/Sample Data')

# Read in and inspect Preprocessed data left by the upstream 'Preprocessing_EDA' component
order_products_prior = pd.read_parquet(files_path / 'target_values.parquet.gz') # Read in historical product purchases data
final_df = pd.read_parquet(files_path / 'merged_products_train.parquet.gz')
print('PREPROCESSED DF:' + '\n', final_df)
print('HISTORICALLY PURCHASED PRODUCTS DF:' + '\n', order_products_prior)

# Use the 'order_products_prior' DF to set the 'target_values'
target_values = order_products_prior.values.tolist()
x_train, x_test, y_train, y_test = train_test_split(final_df, target_values, test_size=0.30, random_state=1)

# Use CountVectorizer for N-Grams feature extraction


# glove-nlp: This is a Python package for working with GloVe embeddings in natural language processing tasks. It includes functions for loading and saving embeddings, 
# as well as tools for text preprocessing and feature extraction.