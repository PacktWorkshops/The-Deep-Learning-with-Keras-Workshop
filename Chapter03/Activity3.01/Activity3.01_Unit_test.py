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
        import Activity3_01
        self.activity = Activity3_01
        
        dirname = self._dirname_if_file('../data/outlier_feats.csv')
        self.feats_loc = os.path.join(dirname, 'outlier_feats.csv')
        self.target_loc = os.path.join(dirname, 'outlier_target.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)
        
        self.seed = 1
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.feats, self.feats)
        pd_testing.assert_frame_equal(self.activity.target, self.target)

    def test_model_1(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model_1 = Sequential()
        model_1.add(Dense(1, activation='sigmoid', input_dim=2)) 
        model_1.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_1.fit(self.feats, self.target, batch_size=5, epochs=100, verbose=0, validation_split=0.2, shuffle=False)
        self.test_loss_1 = model_1.evaluate(self.feats, self.target)
        self.activity_test_loss_1 = self.activity.model_1.evaluate(self.activity.feats, self.activity.target)
        np_testing.assert_almost_equal(self.test_loss_1, self.activity_test_loss_1, decimal=1)

    def test_model_2(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model_2 = Sequential() 
        model_2.add(Dense(3, activation='relu', input_dim=2))
        model_2.add(Dense(1, activation='sigmoid'))
        model_2.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_2.fit(self.feats, self.target, batch_size=5, epochs=200, verbose=0, validation_split=0.2, shuffle=False)
        self.test_loss_2 = model_2.evaluate(self.feats, self.target)
        self.activity_test_loss_2 = self.activity.model_2.evaluate(self.activity.feats, self.activity.target)
        np_testing.assert_almost_equal(self.test_loss_2, self.activity_test_loss_2, decimal=1)

    def test_model_3(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model_3 = Sequential() 
        model_3.add(Dense(6, activation='relu', input_dim=2))
        model_3.add(Dense(1, activation='sigmoid'))
        model_3.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_3.fit(self.feats, self.target, batch_size=5, epochs=400, verbose=0, validation_split=0.2, shuffle=False) 
        self.test_loss_3 = model_3.evaluate(self.feats, self.target)
        self.activity_test_loss_3 = self.activity.model_3.evaluate(self.activity.feats, self.activity.target)
        np_testing.assert_almost_equal(self.test_loss_3, self.activity_test_loss_3, decimal=1)

    def test_model_4(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model_4 = Sequential() 
        model_4.add(Dense(3, activation='tanh', input_dim=2))
        model_4.add(Dense(1, activation='sigmoid'))
        model_4.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_4.fit(self.feats, self.target, batch_size=5, epochs=200, verbose=0, validation_split=0.2, shuffle=False) 
        self.test_loss_4 = model_4.evaluate(self.feats, self.target)
        self.activity_test_loss_4 = self.activity.model_4.evaluate(self.activity.feats, self.activity.target)
        np_testing.assert_almost_equal(self.test_loss_4, self.activity_test_loss_4, decimal=1)

    def test_model_5(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model_5 = Sequential() 
        model_5.add(Dense(6, activation='tanh', input_dim=2))
        model_5.add(Dense(1, activation='sigmoid'))
        model_5.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_5.fit(self.feats, self.target, batch_size=5, epochs=400, verbose=0, validation_split=0.2, shuffle=False) 
        self.test_loss_5 = model_5.evaluate(self.feats, self.target)
        self.activity_test_loss_5 = self.activity.model_4.evaluate(self.activity.feats, self.activity.target)
        np_testing.assert_almost_equal(self.test_loss_5, self.activity_test_loss_5, decimal=1)

if __name__ == '__main__':
    unittest.main()
