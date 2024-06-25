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
models_list = ['logistic_regression', 'RFClassifier', 'XGBoost']

df = pd.read_csv(data_path / 'PreProcessed_ECommerce_Dataset.csv')

# Define target variable (y) and features (X)
X = df.drop(['Churn', 'CashbackAmount'], axis=1)
y = df['Churn']

# Perform Train/Test split with Stratification since the class labels, y, is an imbalanced dataset that favours those who didn't churn (i.e. ~83% didn't churn)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=0.80, 
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=seed)

# Define Hyper Parameters
models_config = {
    'logistic_regression': LogisticRegression(class_weight='balanced', # In addition to Stratification, this perform's class balancing on model at fit() stage
                                              solver='liblinear', # The Solver refers to the available Gradient Descent Optimization options (i.e. selection of Loss functions)
                                              random_state=seed),
    'RFClassifier': RandomForestClassifier(class_weight='balanced',
                                           random_state=seed),
    'XGBoost': xgb.XGBClassifier(device='cuda',
                                 
                                random_state=seed)
}

for model in ['logistic_regression']: # models_config
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