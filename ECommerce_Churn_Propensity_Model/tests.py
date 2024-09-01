import unittest
from pathlib import Path
import pandas as pd
from .src.config import Config

config = Config()
data_path = config.data_path
content_file = config.content_file
float_columns = config.float_cols
categorical_columns = config.categorical_cols
onehot_categoricals = config.onehot_cols
imputed_path = Path(data_path) / 'ECommerce_Dataset_IMPUTED.csv'
preprocessed_path = Path(data_path) / 'PreProcessed_ECommerce_Dataset.csv'

class Test(unittest.TestCase):
    def test_read_impute_data(self):
        """
        Execute Test Case to determine if there are NaNs in the returned
        DataFrame produced by the read_impute_data() function from the 'eda' module.
        """    
        # Load imputed data to run test case
        read_impute_test = pd.read_csv(imputed_path)
        
        # Perform tests and generate pass/fail print statements
        imputation_test = self.assertFalse(expr=read_impute_test.isna().values.any(),
                         msg=f'IMPUTATION TEST CASE FAILED. THERE ARE NANS IN THE \'{imputed_path}\' DATAFRAME!')
        
        if imputation_test is None:
            print(f'IMPUTATION TEST CASE PASSED! THERE ARE NO NANS IN THE \'{imputed_path}\' DATAFRAME')

    def test_preprocessed_data(self):
        """
        Execute Test Case to determine if the preprocessed data from the 'preprocessing' module contains:
        - NaNs: Whilst running the pre_processing() the pd.cut() function's 'right' positional arg may be set to True 
        or not specified. If it's set to either True or not specified, it can create NaNs in the resulting DataFrame. 
        The 'right' arg needs to set to False.
        - renamed columns: The same pd.cut() function will also create a ']' closed column (which is syntactically correct in
        mathematics). Whilst this is mathematically sound, the XGBClassifier() hates it and will raise column name errors in the 
        downstream 'model_training' module. So this 2nd test case will test for existence of these closed columns.
        """
        def bracket_test(df):
            """
            Function to check for the presence of any brackets that are not rounded - i.e. '(' ')'
            """
            for col in df.columns:
                if '[' in col or ']' in col or '{' in col or '}' in col:
                    return True
        
        # Load preprocessed data to run test case
        preprocessed_df_test = pd.read_csv(preprocessed_path)
        
        # Perform tests and generate pass/fail print statements
        renamed_col_test = self.assertFalse(expr=bracket_test(preprocessed_df_test),
                                            msg=f'THE INTERVAL BRACKETS IN \'{preprocessed_path}\' ARE INCOMPATIBLE WITH XGBCLASSIFER(). PLEASE CHECK IF THE RETURNED INTERVAL COLUMNS CONTAIN BRACKETS OTHER THAN \'(\' OR \')\'')

        if renamed_col_test is None:
            print(f'BRACKET SHAPE TEST PASSED! THE RETURNED INTERVAL BRACKETS IN \'{preprocessed_path}\' ARE OF A COMPATIBLE SHAPE WITH XGBCLASSIFER()')

if __name__ == '__main__':
    unittest.main()