import argparse

parser = argparse.ArgumentParser(description='Churn propensity pipeline')
parser.add_argument('--module', type=str, choices=['eda', 'preprocessing', 'model_training'])
args = parser.parse_args()

