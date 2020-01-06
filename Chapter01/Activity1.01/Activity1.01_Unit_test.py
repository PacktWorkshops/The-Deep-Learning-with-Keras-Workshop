import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

class Test(unittest.TestCase):
     
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))

    def setUp(self):
        import Activity1_01
        self.activity = Activity1_01
        
        dirname = self._dirname_if_file('../data/OSI_feats_e3.csv')
        self.feats_loc = os.path.join(dirname, 'OSI_feats_e3.csv')
        self.target_loc = os.path.join(dirname, 'OSI_target_e2.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.feats, self.target, test_size=0.2, random_state=13)
        

        Cs = np.logspace(-2, 6, 9)
        model_l1 = LogisticRegressionCV(Cs=Cs, penalty='l1', cv=10, solver='liblinear', random_state=42, max_iter=10000)
        model_l2 = LogisticRegressionCV(Cs=Cs, penalty='l2', cv=10, random_state=42, max_iter=10000)

        model_l1.fit(self.X_train, self.y_train['Revenue'])
        model_l2.fit(self.X_train, self.y_train['Revenue'])
        
        self.y_pred_l1 = model_l1.predict(self.X_test)
        self.y_pred_l2 = model_l2.predict(self.X_test)
        
        self.accuracy_l1 = metrics.accuracy_score(y_pred=self.y_pred_l1, y_true=self.y_test)
        self.accuracy_l2 = metrics.accuracy_score(y_pred=self.y_pred_l2, y_true=self.y_test)
    
    def test_feats_df(self):
        pd_testing.assert_frame_equal(self.activity.feats, self.feats)
    
    def test_target_df(self):
        pd_testing.assert_frame_equal(self.activity.target, self.target)

    def test_X_train_df(self):
        pd_testing.assert_frame_equal(self.activity.X_train, self.X_train)

    def test_y_train_df(self):
        pd_testing.assert_frame_equal(self.activity.y_train, self.y_train)

    def test_X_test_df(self):
        pd_testing.assert_frame_equal(self.activity.X_test, self.X_test)

    def test_y_test_df(self):
        pd_testing.assert_frame_equal(self.activity.y_test, self.y_test)

    def test_y_pred_l1(self):
        pd_testing.assert_frame_equal(self.activity.y_pred_l1, self.y_pred_l1)

    def test_y_pred_l2(self):
        pd_testing.assert_frame_equal(self.activity.y_pred_l2, self.y_pred_l2)
    
    def test_accuracy_l1(self):
        assert self.activity.accuracy_l1 == self.accuracy_l1

    def test_accuracy_l2(self):
        assert self.activity.accuracy_l2 == self.accuracy_l2

if __name__ == '__main__':
    unittest.main()
