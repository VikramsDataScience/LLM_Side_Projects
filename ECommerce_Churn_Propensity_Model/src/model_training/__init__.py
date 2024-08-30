from ..config import Config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path

# Initialise global variables from config module
config = Config()
data_path = config.data_path
seed = config.seed
churn_app_path = Path(config.churn_app_models)

insample_scores = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score'])
outofsample_scores = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score'])
df = pd.read_csv(Path(data_path) / 'PreProcessed_ECommerce_Dataset.csv')

# Define target variable (y) and features (X)
# The EDA exposed high correlation with the 'CashbackAmount' feature. So remove from X
X = df.drop(['CustomerID', 'Churn', 'CashbackAmount', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
             'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'Tenure_(12, 24)', 'Tenure_(48, 60)',
             'Tenure_(60, 72)', 'PreferredLoginDevice_Computer', 'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone', 'PreferredPaymentMode_CC',
             'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet', 'Gender_Female',
             'Gender_Male', 'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others', 'MaritalStatus_Divorced',
             'MaritalStatus_Married', 'PreferredPaymentMode_Cash on Delivery'], axis=1)
y = df['Churn']

# Define DMatrix and Hyper Parameters for XGBoost
d_matrix = xgb.DMatrix(data=X, 
                       label=y,
                       enable_categorical=True)
params = {
            'objective':'binary:logistic',
            'max_depth': 9,
            # 'alpha': 10, # L1 Regularization on the leaf nodes (larger value means greater regularization) 
            'lambda': 10, # L2 Regularization on the leaf nodes (larger value means greater regularization). L2 is smoother than L1 and tends to better prevent overfitting
            'learning_rate': 0.4,
            'n_estimators':100,
        }

# Perform Train/Test split with Stratification since the class labels, y, is an imbalanced dataset that favours those who didn't churn (i.e. ~83% didn't churn)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=0.80, 
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=seed)

# Define/load Hyper Parameters
models_config = {
    'logistic_regression': LogisticRegression(class_weight='balanced', # In addition to Stratification, this perform's class balancing on model at fit() stage
                                              solver='liblinear', # The Solver refers to the available Gradient Descent Optimization options (i.e. selection of Loss functions)
                                              random_state=seed),
    'RFClassifier': RandomForestClassifier(class_weight=None, # For RFClassifier, setting class_weights='balanced' harms the F1-Score. Leave class_weight=None (default)
                                           n_estimators=200, # ~200 trees improves F1-Score. 200+ is past the point of optimality, and will reduce accuracy
                                           max_depth=None, # Leave max_depth=None so the classifier can grow the trees until all leaves are pure (lowest Gini Impurity) without stopping criterion causing premature terminations
                                           random_state=seed),
    'XGBoost': xgb.XGBClassifier(**params)
}