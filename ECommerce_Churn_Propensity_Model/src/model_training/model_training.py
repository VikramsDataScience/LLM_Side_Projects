import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

# Load variables from __init__.py
from . import models_config, insample_scores, outofsample_scores, X, X_train, X_test, y_train, y_test, Config, data_path

config = Config()
churn_app_path = Path(config.churn_app_models)

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