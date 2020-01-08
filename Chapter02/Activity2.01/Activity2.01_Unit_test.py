import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow import random


class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))
     
    def setUp(self):
        import Activity2_01
        self.activity = Activity2_01
        
        dirname = self._dirname_if_file('../data/OSI_feats.csv')
        self.feats_loc = os.path.join(dirname, 'OSI_feats.csv')
        self.target_loc = os.path.join(dirname, 'OSI_target.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.feats, self.target, test_size=0.2, random_state=42)
        
        np.random.seed(42)
        random.set_seed(42)
        model = Sequential()
        model.add(Dense(1, input_dim=self.X_train.shape[1]))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train['Revenue'], epochs=10, validation_split=0.2, shuffle=False)
        self.test_loss, self.test_acc = model.evaluate(self.X_test, self.y_test['Revenue'])
        
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

    def test_loss(self):
        np_testing.assert_approx_equal(self.activity.test_loss, self.test_loss, significant=2)

    def test_accuracy(self):
        np_testing.assert_approx_equal(self.activity.test_acc, self.test_acc, significant=2)
if __name__ == '__main__':
    unittest.main()
