from os.path import exists
import pandas as pd
from scipy.stats import skew
import numpy as np
from pathlib import Path
from ydata_profiling import ProfileReport
from phik import phik_matrix, significance_matrix
from missforest.missforest import MissForest
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
categorical_columns = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'Complain', 'Churn', 'CouponUsed']
float_columns = ['Tenure', 'WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
skewed_interval_cols = ['WarehouseToHome', 'Tenure', 'CashbackAmount', 'DaySinceLastOrder', 'OrderCount']
interval_bins = {}
missforest_imputer = MissForest()

# Perform count of NaNs in the defined interval columns for downstream bin length calculation using Doane's Formula
# N.B.: NOT NECESSARY IF USING IMPUTATION STRATEGY
skewed_nan_dict = {col: df[col].isna().sum() for col in skewed_interval_cols}
print('SKEWED NaN COUNT DICTIONARY:\n', skewed_nan_dict)

# Cast float_columns as integers and dynamically impute NaN values using MissForest
imputed_df = missforest_imputer.fit_transform(x=df, 
                                   categorical=categorical_columns)
df = pd.DataFrame(imputed_df)
df[float_columns] = df[float_columns].astype(int)
print('\nRECASTED DATA FRAME WITHOUT NaN VALUES:\n', df)

########## Define the Mathematical equations to be used for Skewed bin length calculations ##########
def doanes_formula(data, nan_count) -> int:
    """
    To aid in the preparation for the correct binning of Intervals prior to the calculation of Phi K Correlation,
    I've opted to use Doane's Formula to determine the Bin sizes of the intervals. Since I couldn't find a 
    Python library for the formula, I've come up with this implementation of Doane's Formula.
      Please refer to the 'README.md' file (in the EDA folder) for a mathematical explanation of the formula 
    and the data justifications behind selecting Doane's Formula for calculating bin sizes.
      This function will return the bin length as a truncated integer (not rounded!). I elected for numeric truncation
    over rounding, since I saw that numpy's rounding with np.ceil() led to substantial rounding errors (for instance,
    18.1 would be rounded to 19). So, I've opted to truncate and cast as integer rather than have these rounding 
    errors in the calculation of the interval's bin length.
      N.B.: Since Doane's Formula relies on the number of observations in the DF, if there are NaNs in the input DF, 
    please calculate the number of NaNs and deduct that value from 'n' (i.e. use the 'nan_count' arg in the function).
    If there are no NaNs, please set 'nan_count = 0'.
    """
    n = len(data) - nan_count
    g1 = skew(data) - nan_count
    sigma_g1 = np.sqrt((6*(n - 2)) / ((n + 1)*(n + 3)))
    k = 1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1)

    return int(np.trunc(k))

########## Phi K Correlation calculation and report generation ##########
# Apply Doane's Formula to calculate and store bin sizes for the skewed data in a Dictionary structure as prepartion for Phi K Correlation
for col in skewed_interval_cols:
    skewed_bin_len = doanes_formula(df[col], nan_count=0) # skewed_nan_dict[col]
    intervals = {
    col: skewed_bin_len
    }
    interval_bins.update(intervals)

print('RESULTS OF DOANE\'S CALCULATION OF BIN LENGTHS FOR DECLARED INTERVAL VARIABLES:\n', interval_bins)

# If the following Matrices don't exist, generate and store them as CSVs
if not exists(Path(data_path) / 'phi_k_matrix.csv') or not exists(Path(data_path) / 'significance_matrix.csv'):

    phik_matrix(df, 
                bins=interval_bins, 
                interval_cols=interval_bins,
                noise_correction=True).to_csv(Path(data_path) / 'phi_k_matrix.csv')
    
    # Please note that calculating a Significance Matrix can be a little slow!
    significance_matrix(df,
                        bins=interval_bins,
                        interval_cols=interval_bins,
                        significance_method='hybrid' # Hybrid method between calculating G-Test Statistic (asymptotic) and Monte Carlo simulations is default and recommended by the authors
                        ).to_csv(Path(data_path) / 'significance_matrix.csv')
    
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
                                                            'warn_high_correlations': True}},
                                    tsmode=False, 
                                    explorative=True)

    profile_report.to_file(Path(data_path) / 'EDA_Profiling_Report.html')