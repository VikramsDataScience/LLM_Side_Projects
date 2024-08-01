from flask import Flask, request, jsonify, render_template
from pathlib import Path
from .config import Config
import pickle
from statistics import median

# Load the file paths and global variables from the Config file
config = Config()
churn_app_path = Path(config.churn_app_models)
models_list = config.models_list
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