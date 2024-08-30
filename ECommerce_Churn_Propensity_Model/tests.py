import unittest
from pathlib import Path
from .src.config import Config
from .src.eda import read_impute_data
from .src.preprocessing import pre_processing

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
        DataFrame produced by the read_impute_data() function.
        IMPORTANT NOTE: Please update the 'imputed_path' object as needed when using
        this Test Case!
        """        
        read_impute_test = read_impute_data(df_path=Path(content_file), 
                                            sheet_name=1,
                                            float_cols=float_columns, 
                                            categorical_cols=categorical_columns,
                                            output_path=imputed_path)
        
        self.assertFalse(expr=read_impute_test.isna().values.any(),
                         msg=f'IMPUTATION TEST CASE FAILED. THERE ARE NANS IN THE \'{imputed_path}\' DATAFRAME!')
        
        print(f'IMPUTATION TEST CASE PASSED! THERE ARE NO NANS IN THE \'{imputed_path}\' DATAFRAME')

    def test_preprocessed_data(self):
        """
        Execute Test Case to determine if the preprocessed data contains:
        - NaNs: Whilst running the pre_processing() the pd.cut() function's 'right' positional arg may be set to True 
        or not specified. If it's set to either True or not specified, it can create NaNs in the resulting DataFrame. 
        The 'right' arg needs to set to False.
        - renamed columns: The same pd.cut() function will also create a ']' closed column (which is mathematically
        very correct!). Whilst this is mathematically sound, the XGBClassifier() hates it and will raise column name 
        errors in the downstream 'model_training' module. So the 2nd test case will test for existence of these closed columns.
        IMPORTANT NOTE: Please update the 'preprocessed_path' object as needed when using
        this Test Case!
        """
        def bracket_test(df):
            """
            Function to check for the presence of any brackets that are not rounded - i.e. '(' ')'
            """
            for col in df.columns:
                if '[' in col or ']' in col or '{' in col or '}' in col:
                    return True
        
        pre_process_test = pre_processing(df_path=imputed_path,
                                        bins=[0, 12, 24, 48, 60, 72], 
                                        onehot_cols=onehot_categoricals, 
                                        output_path=preprocessed_path, 
                                        bin_cols='Tenure')
        
        # Perform tests and generate pass/fail print statements
        nan_test = self.assertFalse(expr=pre_process_test.isna().values.any(),
                                    msg='THERE ARE NANS IN THE INTERVAL COLUMNS! PLEASE CHECK THE PD.CUT() \'RIGHT\' ARG TO DEBUG')
        renamed_col_test = self.assertFalse(expr=bracket_test(pre_process_test),
                                            msg='THE INTERVAL BRACKETS ARE INCOMPATIBLE WITH XGBCLASSIFER(). PLEASE CHECK IF THE RETURNED INTERVAL COLUMNS CONTAIN BRACKETS OTHER THAN \'(\' OR \')\'')
        
        if nan_test is None:
            print('NAN TEST PASSED! THERE ARE NO NANS IN THE PREPROCESSED DATAFRAME')
        if renamed_col_test is None:
            print('BRACKET SHAPE TEST PASSED! THE RETURNED INTERVAL BRACKETS ARE OF A COMPATIBLE SHAPE WITH XGBCLASSIFER()')

if __name__ == '__main__':
    unittest.main()