from os.path import exists
from pathlib import Path
from ydata_profiling import ProfileReport
from phik import phik_matrix, significance_matrix

# Load variables from __init__.py
from . import read_impute_data, doanes_formula, Config

# Load the file paths and global variables from the Config file
config = Config()
content_file = config.content_file
data_path = config.data_path
categorical_columns = config.categorical_cols
skewed_interval_columns = config.skewed_interval_cols
float_columns = config.float_cols
interval_bins = {}

df = read_impute_data(df_path=Path(content_file), sheet_name=1, float_cols=float_columns, categorical_cols=categorical_columns)
df[float_columns] = df[float_columns].astype(int)
print('\nRECASTED DATA FRAME WITHOUT NaN VALUES:\n', df)

########## Phi K Correlation calculation and report generation ##########
# Apply Doane's Formula to calculate and store bin sizes for the skewed data in a Dictionary structure as prepartion for Phi K Correlation
for col in skewed_interval_columns:
    skewed_bin_len = doanes_formula(df[col], nan_count=0)
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