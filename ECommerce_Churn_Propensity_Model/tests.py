import unittest
from pathlib import Path
from .src import read_impute_data, pre_processing, Config

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
        try:
            read_impute_test = read_impute_data(df_path=Path(content_file), 
                                                sheet_name=1,
                                                float_cols=float_columns, 
                                                categorical_cols=categorical_columns,
                                                output_path=imputed_path)
            
            self.assertFalse(expr=read_impute_test.isna().values.any())
            print(f'IMPUTATION TEST CASE PASSED! THERE ARE NO NANS IN THE \'{imputed_path}\' DATAFRAME')

        except AssertionError:
            print(f'IMPUTATION TEST CASE FAILED. THERE ARE NANS IN THE \'{imputed_path}\' DATAFRAME!')
            raise

if __name__ == '__main__':
    unittest.main()