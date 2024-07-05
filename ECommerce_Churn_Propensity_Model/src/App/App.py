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
	    data['CityTier'],
        data['WarehouseToHome'],
        data['HourSpendOnApp'],
        data['NumberOfDeviceRegistered'],
        data['SatisfactionScore'],
        data['NumberOfAddress'],
        data['Complain'],
        data['OrderAmountHikeFromlastYear'],
        data['CouponUsed'],
        data['OrderCount'],
        data['DaySinceLastOrder'],
        data['Tenure_(0, 12)'],
        data['Tenure_(12, 24)'],
        data['Tenure_(24, 48)'],
        data['PreferredLoginDevice_Computer'],
        data['PreferredLoginDevice_Mobile Phone'],
        data['PreferredLoginDevice_Phone'],
        data['PreferredPaymentMode_CC'],
        data['PreferredPaymentMode_COD'],
        data['PreferredPaymentMode_Cash on Delivery'],
        data['PreferredPaymentMode_Credit Card'],
        data['PreferredPaymentMode_Debit Card'],
        data['PreferredPaymentMode_E wallet'],
        data['PreferredPaymentMode_UPI'],
        data['Gender_Female'],
        data['Gender_Male'],
        data['PreferedOrderCat_Fashion'],
        data['PreferedOrderCat_Grocery'],
        data['PreferedOrderCat_Laptop & Accessory'],
        data['PreferedOrderCat_Mobile Phone'],
        data['PreferedOrderCat_Others'],
        data['MaritalStatus_Divorced'],
        data['MaritalStatus_Married'],
        data['MaritalStatus_Single']
    ]

    # Select positive class (i.e. probability of churn). Shape of predict_proba() should be [negative class, positive class]
    log_reg_pred = log_reg_model.predict_proba([features])[0][1]
    RF_pred = RF_model.predict_proba([features])[0][1]
    XG_pred = XGBoost_model.predict_proba([features])[0][1]

    combined_predictions = [log_reg_pred, RF_pred, XG_pred]

    # Since Logistic Regression has significanly lower accuracy than RF and XGBoost, calculate and return the median F1-Score across all three models
    return jsonify({'prediction': median(combined_predictions)})

if __name__ == '__main__':
    app.run(debug=True)