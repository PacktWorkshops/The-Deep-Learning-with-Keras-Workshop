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
        import Exercise1_03
        self.exercise = Exercise1_03
        
        dirname = self._dirname_if_file('../data/OSI_feats_e2.csv')
        self.data_loc = os.path.join(dirname, 'OSI_feats_e2.csv')
        
        self.data = pd.read_csv(self.data_loc)
        
        operation_system_dummies = pd.get_dummies(self.data['OperatingSystems'], prefix='OperatingSystems')
        operation_system_dummies.drop('OperatingSystems_5', axis=1, inplace=True)
        self.data = pd.concat([self.data, operation_system_dummies], axis=1)

        browser_dummies = pd.get_dummies(self.data['Browser'], prefix='Browser')
        browser_dummies.drop('Browser_9', axis=1, inplace=True)
        self.data = pd.concat([self.data, browser_dummies], axis=1)
        
        traffic_dummies = pd.get_dummies(self.data['TrafficType'], prefix='TrafficType')
        traffic_dummies.drop('TrafficType_17', axis=1, inplace=True)
        self.data = pd.concat([self.data, traffic_dummies], axis=1)

        region_dummies = pd.get_dummies(self.data['Region'], prefix='Region')
        region_dummies.drop('Region_5', axis=1, inplace=True)
        self.data = pd.concat([self.data, region_dummies], axis=1)
        drop_cols = ['OperatingSystems', 'Browser', 'TrafficType', 'Region']
        self.data.drop(drop_cols, inplace=True, axis=1)                 
                                              
    def test_data_df(self):
        pd_testing.assert_frame_equal(self.exercise.data, self.data)


if __name__ == '__main__':
    unittest.main()
