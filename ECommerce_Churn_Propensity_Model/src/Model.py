import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path
from .config import Config
import pickle

# Load the file paths and global variables from the Config file
config = Config()
data_path = Path(config.data_path)
churn_app_path = Path(config.churn_app_models)
seed = config.seed

insample_scores = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score'])
outofsample_scores = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score'])
df = pd.read_csv(data_path / 'PreProcessed_ECommerce_Dataset.csv')

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

for model in models_config:
    print(f'COMMENCING TRAINING FOR \'{model}\':')

    # Train set predictions (In sample)
    models_config[model].fit(X_train, y_train)
    y_pred = models_config[model].predict(X_train)
    train_classification_scores = precision_recall_fscore_support(y_train, y_pred, average='weighted')
    # Save in-sample Prediction scores
    new_row = pd.DataFrame({
    'Model': [model],
    'Precision': [train_classification_scores[0]],
    'Recall': [train_classification_scores[1]],
    'F1-Score': [train_classification_scores[2]]
    })
    insample_scores = pd.concat([insample_scores, new_row], ignore_index=True)

    train_conf_matrix = confusion_matrix(y_train, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=train_conf_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.savefig(Path(data_path / 'train_conf_matrix.png'))
    
    # Test set predictions (Out of Sample)
    y_test_pred = models_config[model].predict(X_test)
    test_classification_scores = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
    # Save out-of-sample Prediction scores
    new_row = pd.DataFrame({
    'Model': [model],
    'Precision': [test_classification_scores[0]],
    'Recall': [test_classification_scores[1]],
    'F1-Score': [test_classification_scores[2]]
    })
    outofsample_scores = pd.concat([outofsample_scores, new_row], ignore_index=True)

    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.savefig(Path(data_path / 'OOS_conf_matrix.png'))

    if model == 'logistic_regression':
        print(f'{model} Feature Importances:\n', models_config[model].coef_)
    if model == 'RFClassifier':
        rf_feat_importance = models_config[model].feature_importances_
        feat_importance = pd.Series(rf_feat_importance, index=X.columns)
        print(f'{model} Feature Importances:\n', feat_importance)
    if model == 'XGBoost':
        xg_feat_importance = models_config[model].get_booster().get_score(importance_type="gain")
        print(f'{model} Feature Importances:\n', xg_feat_importance)
    
    # Save models for downstream consumption
    with open(churn_app_path / f'churn_{model}_model.pkl', 'wb') as file:
        pickle.dump(models_config[model], file)

# Save prediction scores to storage
insample_scores.to_csv(Path(data_path) / 'insample_scores.csv')
outofsample_scores.to_csv(Path(data_path) / 'outofsample_scores.csv')