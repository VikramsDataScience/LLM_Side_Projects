from os.path import exists
import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport
import phik
from phik import report
import yaml

# Load the file paths and global variables from YAML config file
try:
    config_path = Path('C:/Users/Vikram Pande/Side_Projects_(OUTSIDE_REPO)/ECommerce_Churn_Propensity_Model')

    with open(config_path / 'config.yml', 'r') as file:
        global_vars = yaml.safe_load(file)
except:
    print(f'{config_path} YAML Configuration file path not found. Please check the storage path of the \'config.yml\' file and try again')

content_file = global_vars['content_path']
data_path = global_vars['data_path']

df = pd.read_excel(Path(content_file), sheet_name=1)
print(df)

# If the EDA profiling report doesn't exist, generate report as an HTML document
if not exists(Path(data_path) / 'EDA_Profiling_Report.html'):
    # Explicitly declare categorical variables to assist y-data's report generation (everything else in the dataframe y-data can infer)
    df_schema = {'Churn': 'categorical',
                'Complain': 'categorical',
                'CouponUsed': 'categorical',
                'Gender': 'categorical',
                'MaritalStatus': 'categorical',
                'PreferredLoginDevice': 'categorical',
                'PreferredPaymentMode': 'categorical',
                'PreferedOrderCat': 'categorical'}

    profile_report = ProfileReport(df, 
                                    title='ECommerce Customer Churn EDA Report', 
                                    type_schema=df_schema,
                                    correlations={'phi_k': {'calculate': True,
                                                            'threshold': 0.5,
                                                            'warn_high_correlations': True}}, # Since there are 10 categorical variables, use Phi K to calculate correlations between them. Correlations >=0.5 carry high risk for modelling
                                    tsmode=False, 
                                    explorative=True)

    profile_report.to_file(Path(data_path) / 'EDA_Profiling_Report.html')

# If the Phi K Correlation report doesn't exist, generate the PDF
if not exists(Path(data_path) / 'phi_k_report.pdf'):
    # Define schema
    df_schema = {'Churn': 'categorical',
                'Complain': 'categorical',
                'CouponUsed': 'categorical',
                'Gender': 'categorical',
                'MaritalStatus': 'categorical',
                'PreferredLoginDevice': 'categorical',
                'PreferredPaymentMode': 'categorical',
                'PreferedOrderCat': 'categorical',
                'SatisfactionScore': 'ordinal',
                'CityTier': 'ordinal'}
    
    phik.phik_matrix(df, )
    report.correlation_report(df, 
                              correlation_threshold=0.5,
                              pdf_file_name=Path(data_path) / 'phi_k_report.pdf')