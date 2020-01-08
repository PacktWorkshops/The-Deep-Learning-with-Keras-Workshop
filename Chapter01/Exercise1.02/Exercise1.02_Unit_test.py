import unittest
import pandas as pd
import pandas.testing as pd_testing
import numpy.testing as np_testing
import os
import import_ipynb


class Test(unittest.TestCase):
     
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))

    def setUp(self):
        import Exercise1_02
        self.exercise = Exercise1_02
        
        dirname = self._dirname_if_file('../data/OSI_feats.csv')
        self.feats_loc = os.path.join(dirname, 'OSI_feats.csv')
        self.target_loc = os.path.join(dirname, 'OSI_target.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)

        self.feats['is_weekend'] = self.feats['Weekend'].apply(lambda row: 1 if row == True else 0)

        visitor_type_dummies = pd.get_dummies(self.feats['VisitorType'], prefix='VisitorType')
        visitor_type_dummies.drop('VisitorType_Other', axis=1, inplace=True)
        self.feats = pd.concat([self.feats, visitor_type_dummies], axis=1)
        
        month_dummies = pd.get_dummies(self.feats['Month'], prefix='Month')
        month_dummies.drop('Month_Feb', axis=1, inplace=True)
        self.feats = pd.concat([self.feats, month_dummies], axis=1)
        self.feats.drop(['Weekend', 'VisitorType', 'Month'], axis=1, inplace=True, errors='ignore')                 
                                          
        self.target['Revenue'] = self.target['Revenue'].apply(lambda row: 1 if row == True else 0)
    
    def test_feats_df(self):
        pd_testing.assert_frame_equal(self.exercise.data, self.feats)

    def test_target_df(self):
        pd_testing.assert_frame_equal(self.exercise.target, self.target)

if __name__ == '__main__':
    unittest.main()
