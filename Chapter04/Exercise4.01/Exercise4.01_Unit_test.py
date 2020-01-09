import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb

class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Exercise4_01
        self.exercise = Exercise4_01
        
        dirname = self._dirname_if_file('../data/qsar_fish_toxicity.csv')
        self.data_loc = os.path.join(dirname, 'qsar_fish_toxicity.csv')
        
        colnames = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC','MLOGP', 'LC50']
        self.data = pd.read_csv(self.data_loc, sep=';', names=colnames)
        self.X = self.data.drop('LC50', axis=1)
        self.y = self.data['LC50']
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.exercise.X, self.X)
        pd_testing.assert_series_equal(self.exercise.y, self.y)


if __name__ == '__main__':
    unittest.main()
