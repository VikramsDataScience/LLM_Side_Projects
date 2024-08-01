import os
import sys
import contextlib

class Config:
    def __init__(self):
        self.model_ver = 0.01
        self.seed = 314
        self.content_path = 'C:/Sample Data/Ecommerce_Churn_Data/ECommerce_Dataset.xlsx'
        self.data_path = 'C:/Sample Data/Ecommerce_Churn_Data'
        self.churn_app_models = 'C:/Sample Data/Ecommerce_Churn_Data/churn_app/models'
        self.models_list = ['logistic_regression', 'RFClassifier', 'XGBoost']
        self.float_cols = ['Tenure', 'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
        self.categorical_cols = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'Complain', 'Churn', 'CouponUsed']
        self.onehot_cols = ['Tenure', 'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']

@contextlib.contextmanager
def suppress_stdout():
    """
    For any library that contains (undesirably) verbose output, use this boilerplate function to suppress
    that output in the CLI.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
