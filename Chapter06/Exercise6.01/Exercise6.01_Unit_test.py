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
        import Exercise6_01
        self.exercise = Exercise6_01
        
        dirname = self._dirname_if_file('../data/pacific_hurricanes.csv')
        self.df_loc = os.path.join(dirname, 'pacific_hurricanes.csv')
        
        self.df = pd.read_csv(self.df_loc)
        
    def test_input_frame(self):
        pd_testing.assert_frame_equal(self.exercise.df, self.df)

 
        
if __name__ == '__main__':
    unittest.main()
