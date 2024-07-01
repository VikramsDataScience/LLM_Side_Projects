from flask import Flask, request, jsonify, render_template
from pathlib import Path
import yaml
import pickle

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

churn_app_path = Path(global_vars['churn_app_models'])
models_list = ['logistic_regression', 'RFClassifier', 'XGBoost']
app = Flask(__name__)

# Load pickled models saved by the upstream 'Model' module
for model in models_list:
    with open(churn_app_path / f'churn_{model}_model.pkl', 'rb') as file:
        if model == 'logistic_regression':
            log_reg_model = pickle.load(file)
        if model == 'RFClassifier':
            RF_model = pickle.load(file)
        if model == 'XGBoost':
            XGBoost_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

