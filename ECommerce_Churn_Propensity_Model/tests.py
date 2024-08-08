import unittest
from pathlib import Path
from .src import read_impute_data, pre_processing, Config

config = Config()
data_path = config.data_path
content_file = config.content_file
float_columns = config.float_cols
categorical_columns = config.categorical_cols
onehot_categoricals = config.onehot_cols

class Test(unittest.TestCase):
    def test_read_impute_data(self):
        """
        Execute Test Case to determine if there are NaNs in the returned
        DataFrame produced by the read_impute_data() function.
        """
        read_impute_test = read_impute_data(df_path=Path(content_file), 
                      sheet_name=1, 
                      float_cols=float_columns, 
                      categorical_cols=categorical_columns)
        self.assertFalse(read_impute_test.isna())

if __name__ == '__main__':
    unittest.main()