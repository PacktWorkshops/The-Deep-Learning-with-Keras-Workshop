import unittest
import import_ipynb
import pandas as pd
import pandas.testing as pd_testing
import numpy.testing as np_testing
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class Test(unittest.TestCase):
     
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))

    def setUp(self):
        import Exercise1_05
        self.exercise = Exercise1_05
        
        dirname = self._dirname_if_file('../data/OSI_target_e2.csv')
        self.target_loc = os.path.join(dirname, 'OSI_target_e2.csv')
        
        self.target = pd.read_csv(self.target_loc)        
                       
        self.y_baseline = pd.Series(data=[0]*self.target.shape[0])
        
        self.precision, self.recall, self.fscore, _ = metrics.precision_recall_fscore_support(
            y_pred=self.y_baseline, y_true=self.target['Revenue'], average='macro', zero_division=1)
    
    def test_target_df(self):
        pd_testing.assert_frame_equal(self.exercise.target, self.target)

    def test_precision(self):
        assert self.exercise.precision == self.precision

    def test_recall(self):
        assert self.exercise.recall == self.recall

    def test_fscore(self):
        assert self.exercise.fscore == self.fscore

if __name__ == '__main__':
    unittest.main()
