import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path
import yaml

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

data_path = Path(global_vars['data_path'])
seed = global_vars['seed']

df = pd.read_csv(data_path / 'PreProcessed_ECommerce_Dataset.csv')

# Define target variable (y) and features (X)
# The EDA exposed high correlation with the 'CashbackAmount' feature. So remove from X
X = df.drop(['CustomerID', 'Churn', 'CashbackAmount'], axis=1)
y = df['Churn']

# Define DMatrix and Hyper Parameters for XGBoost
d_matrix = xgb.DMatrix(data=X, label=y)
params = {
            'objective':'binary:logistic',
            'max_depth': 9,
            # 'alpha': 10, # L1 Regularization on the leaf nodes (larger value means greater regularization) 
            'lambda': 10, # L2 Regularization on the leaf nodes (larger value means greater regularization). L2 is smoother than L1 and tends to better prevent overfitting
            'learning_rate': 0.4,
            'n_estimators':100
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
    print(f'In sample prediction scores for {model}:')
    print('Precision: ', train_classification_scores[0],
          '\nRecall: ', train_classification_scores[1],
          '\nF1-Score: ', train_classification_scores[2])
    train_conf_matrix = confusion_matrix(y_train, y_pred)
    print(train_conf_matrix)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=train_conf_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.savefig(Path(data_path / 'train_conf_matrix.png'))
    
    # Test set predictions (Out of Sample)
    y_test_pred = models_config[model].predict(X_test)
    test_classification_scores = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
    print(f'\nOut of sample prediction scores for {model}:')
    print('Precision: ', test_classification_scores[0],
          '\nRecall: ', test_classification_scores[1],
          '\nF1-Score: ', test_classification_scores[2])
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    print(test_conf_matrix)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.savefig(Path(data_path / 'OOS_conf_matrix.png'))