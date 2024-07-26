from flask import Flask, request, jsonify, render_template
from pathlib import Path
import yaml
import pickle
from statistics import median

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = [
        int(data['SatisfactionScore']),
        int(data['Complain']),
        int(data['Tenure_(0, 12)']),
        int(data['Tenure_(24, 48)']),
        int(data['PreferredPaymentMode_COD']),
        int(data['PreferredPaymentMode_UPI']),
        int(data['PreferedOrderCat_Fashion']),
        int(data['PreferedOrderCat_Grocery']),
        int(data['PreferedOrderCat_Laptop & Accessory']),
        int(data['MaritalStatus_Single'])
    ]

    # Select positive class (i.e. probability of churn). Shape of predict_proba() should be [negative class, positive class]
    log_reg_pred = log_reg_model.predict_proba([features])[0][1]
    RF_pred = RF_model.predict_proba([features])[0][1]
    XG_pred = XGBoost_model.predict_proba([features])[0][1]

    combined_predictions = [float(log_reg_pred), float(RF_pred), float(XG_pred)]

    # Since Logistic Regression has significanly lower accuracy than RF and XGBoost, calculate and return the median F1-Score across all three models
    return jsonify({'prediction': f'{round(median(combined_predictions) * 100, ndigits=2)}%'})

if __name__ == '__main__':
    app.run(debug=True)