class Config:
    def __init__(self):
        self.seed = 314
        self.content_file = 'ECommerce_Churn_Propensity_Model/data/ECommerce_Dataset.xlsx' # 'C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model/data/ECommerce_Dataset.xlsx'
        self.data_path = 'ECommerce_Churn_Propensity_Model/data' # 'C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model/data'
        self.churn_app_models = 'C:/Sample Data/Ecommerce_Churn_Data/churn_app/models'
        self.models_list = ['logistic_regression', 'RFClassifier', 'XGBoost']
        self.float_cols = ['Tenure', 'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
        self.categorical_cols = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'Complain', 'Churn', 'CouponUsed']
        self.onehot_cols = ['Tenure', 'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
        self.skewed_interval_cols = ['WarehouseToHome', 'Tenure', 'CashbackAmount', 'DaySinceLastOrder', 'OrderCount']