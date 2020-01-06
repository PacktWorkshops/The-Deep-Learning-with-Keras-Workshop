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
        import Exercise1_04
        self.exercise = Exercise1_04
        
        dirname = self._dirname_if_file('../data/OSI_feats_e3.csv')
        self.feats_loc = os.path.join(dirname, 'OSI_feats_e3.csv')
        self.target_loc = os.path.join(dirname, 'OSI_target_e2.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)        
        
        test_size = 0.2
        random_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.feats, self.target, test_size=test_size, random_state=random_state)
        model = LogisticRegression(random_state=42, max_iter=10000)
        model.fit(self.X_train, self.y_train['Revenue'])                 
        self.y_pred = model.predict(self.X_test)
        
        self.accuracy = metrics.accuracy_score(y_pred=self.y_pred, y_true=self.y_test)
    
    def test_feats_df(self):
        pd_testing.assert_frame_equal(self.exercise.feats, self.feats)

    def test_target_df(self):
        pd_testing.assert_frame_equal(self.exercise.target, self.target)

    def test_X_train_df(self):
        pd_testing.assert_frame_equal(self.exercise.X_train, self.X_train)

    def test_y_train_df(self):
        pd_testing.assert_frame_equal(self.exercise.y_train, self.y_train)

    def test_X_test_df(self):
        pd_testing.assert_frame_equal(self.exercise.X_test, self.X_test)

    def test_y_test_df(self):
        pd_testing.assert_frame_equal(self.exercise.y_test, self.y_test)

    def test_accuracy(self):
        assert self.exercise.accuracy == self.accuracy

if __name__ == '__main__':
    unittest.main()
