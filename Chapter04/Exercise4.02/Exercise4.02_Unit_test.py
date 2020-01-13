import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from tensorflow import random

def build_model():
    model = Sequential()
    model.add(Dense(8, input_dim=6, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Exercise4_02
        self.exercise = Exercise4_02
        
        dirname = self._dirname_if_file('../data/qsar_fish_toxicity.csv')
        self.data_loc = os.path.join(dirname, 'qsar_fish_toxicity.csv')
        
        colnames = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC','MLOGP', 'LC50']
        self.data = pd.read_csv(self.data_loc, sep=';', names=colnames)
        self.X = self.data.drop('LC50', axis=1)
        self.y = self.data['LC50']
        self.seed = 1
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.exercise.X, self.X)
        pd_testing.assert_series_equal(self.exercise.y, self.y)

    def test_model_perf(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model = KerasRegressor(build_fn= build_model, epochs=100, batch_size=20, verbose=0)
        kf = KFold(n_splits=5)
        self.results = cross_val_score(model, self.X, self.y, cv=kf)
        np_testing.assert_array_almost_equal(self.exercise.results, self.results, decimal=0)


if __name__ == '__main__':
    unittest.main()
