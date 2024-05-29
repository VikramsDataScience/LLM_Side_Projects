from os.path import exists
import pandas as pd
from scipy.stats import skew
import numpy as np
from pathlib import Path
from ydata_profiling import ProfileReport
from phik import report, phik_matrix, significance_matrix
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

# Define columns for casting and interval definitions
float_columns = ['Tenure', 'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
interval_cols = ['WarehouseToHome', 'Tenure']
interval_dicts = {}

# Cast float_columns as integers and impute NaN values with 0s
df[float_columns] = df[float_columns].fillna(0).astype(int)
print(df)

def doanes_formula(data) -> int:
    """
    To aid in the preparation for the correct binning of Intervals prior to the calculation of Phi K Correlation,
    I've opted to use Doane's Formula to determine the Bin sizes of the intervals. Since I couldn't find a 
    Python library for the formula, I've come up with this implementation of Doane's Formula.
      Please refer to the 'README.md' file (in the EDA folder) for a mathematical explanation of the formula 
    and the data justifications behind selecting Doane's Formula for calculating bin sizes.
      This function will return the bin length as a truncated integer.
    """
    n = len(data)
    g1 = skew(data)
    sigma_g1 = np.sqrt((6*(n - 2)) / ((n + 1)*(n + 3)))
    k = 1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1)

    return int(np.trunc(k))

########## Y-Data Profiling ##########
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
                                                            'warn_high_correlations': True}}, # Since there are 10 categorical variables, use Phi K to calculate correlations between them
                                    tsmode=False, 
                                    explorative=True)

    profile_report.to_file(Path(data_path) / 'EDA_Profiling_Report.html')

########## Phi K Correlation calculation and report generation ##########
# Apply Doane's Formula to calculate and store bin sizes in a List structure as prepartion for Phi K Correlation
for col in interval_cols:
    bin_len = doanes_formula(df[col])
    intervals = {
    col: bin_len
    }

    interval_dicts.update(intervals)

print(interval_dicts)

# If the Phi K Correlation report or the Phi K Correlation Matrix doesn't exist, generate the PDF
if not exists(Path(data_path) / 'phi_k_report.pdf') or not exists(Path(data_path) / 'phi_k_matrix.csv') or not exists(Path(data_path) / 'significance_matrix.csv'):

    phik_matrix(df, 
                bins=interval_dicts, 
                interval_cols=interval_cols,
                noise_correction=True).to_csv(Path(data_path) / 'phi_k_matrix.csv')
    
    significance_matrix(df,
                        bins=interval_dicts,
                        interval_cols=interval_cols,
                        significance_method='hybrid' # Hybrid method between calculating G-Test Statistic (asymptotic) and Monte Carlo simulations is default and recommended by the authors
                        ).to_csv(Path(data_path) / 'significance_matrix.csv')
    
    report.correlation_report(df,
                              interval_cols=interval_cols,
                              correlation_threshold=0.5, # In Phi K, correlations >=0.5 carry high risk for modelling
                              pdf_file_name=Path(data_path) / 'phi_k_report.pdf')