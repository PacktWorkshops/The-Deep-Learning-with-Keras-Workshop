import unittest
import import_ipynb
import pandas as pd
import pandas.testing as pd_testing
import numpy.testing as np_testing
import os

class Test(unittest.TestCase):
     
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))

    def setUp(self):
        import Exercise1_01
        self.exercise = Exercise1_01
        
        dirname = self._dirname_if_file('../data/online_shoppers_intention.csv')
        self.file_loc = os.path.join(dirname, 'online_shoppers_intention.csv')
        self.data = pd.read_csv(self.file_loc)
        
        self.feats = self.data.drop('Revenue', axis=1)
        self.target = self.data['Revenue']
 
    def test_data_df(self):
        pd_testing.assert_frame_equal(self.exercise.data, self.data)

    def test_feats_df(self):
        pd_testing.assert_frame_equal(self.exercise.feats, self.feats)

    def test_target_df(self):
        pd_testing.assert_series_equal(self.exercise.target, self.target)

if __name__ == '__main__':
    unittest.main()
