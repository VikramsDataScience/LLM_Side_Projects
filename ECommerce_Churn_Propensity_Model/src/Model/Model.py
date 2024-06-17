import pandas as pd
from sklearn.model_selection import train_test_split
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
X = df.drop(['Churn'], axis=1)
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
    'logistic_regression': LogisticRegression(class_weight='balanced', # Even with stratification performed on the class labels during train_test_split(), set 'class_weight'='balanced'
                                              random_state=seed),
    'RFClassifier': RandomForestClassifier(class_weight='balanced',
                                            random_state=seed),
    'XGBoost': xgb.XGBClassifier(device='cuda',

    )
}

for model in models_list:
    print(f'COMMENCING TRAINING FOR {model}:\n')

    models_config[model].fit(X_train, y_train)
    y_pred = models_config[model].predict(X_test)
