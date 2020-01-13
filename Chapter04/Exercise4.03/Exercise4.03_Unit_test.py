import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import import_ipynb
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow import random
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def build_model_1():
    model = Sequential()
    model.add(Dense(4, input_dim=6, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_model_2():
    model = Sequential()
    model.add(Dense(8, input_dim=6, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_model_3():
    model = Sequential()
    model.add(Dense(4, input_dim=6, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_model_2_mod(activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(8, input_dim=6, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Exercise4_03
        self.exercise = Exercise4_03
        
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

    def test_model_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        self.results_1 = []
        models = [build_model_1, build_model_2, build_model_3]
        # loop over three models
        for m in range(len(models)):
            model = KerasRegressor(build_fn=models[m], epochs=100, batch_size=20, verbose=0, shuffle=False)
            kf = KFold(n_splits=3)
            result = cross_val_score(model, self.X, self.y, cv=kf)
            self.results_1.append(result)
            
        np_testing.assert_array_almost_equal(
            np.array(self.results_1).mean(axis=1), np.array(self.exercise.results_1).mean(axis=1), decimal=0)
        np_testing.assert_array_almost_equal(
            np.array(self.results_1).std(axis=1), np.array(self.exercise.results_1).std(axis=1), decimal=0)
        
        

    def test_batch_epoch_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        self.results_2 = []
        epochs = [100, 150]
        batches = [20, 15]

        # Loop over pairs of epochs and batch_size
        for e in range(len(epochs)):
            for b in range(len(batches)):
                model = KerasRegressor(build_fn= build_model_2, epochs= epochs[e], batch_size= batches[b], verbose=0, shuffle=False)
                kf = KFold(n_splits=3)
                result = cross_val_score(model, self.X, self.y, cv=kf)
                self.results_2.append(result)
 
        np_testing.assert_array_almost_equal(
            np.array(self.results_2).mean(axis=1), np.array(self.exercise.results_2).mean(axis=1), decimal=0)
        np_testing.assert_array_almost_equal(
            np.array(self.results_2).std(axis=1), np.array(self.exercise.results_2).std(axis=1), decimal=0)

    def test_opt_act_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        self.results_3 = []
        activations = ['relu', 'tanh']
        optimizers = ['sgd', 'adam', 'rmsprop']
        for o in range(len(optimizers)):
            for a in range(len(activations)):
                optimizer = optimizers[o]
                activation = activations[a]
                model = KerasRegressor(build_fn= build_model_2_mod, epochs=100, batch_size=20, verbose=0, shuffle=False)
                kf = KFold(n_splits=3)
                result = cross_val_score(model, self.X, self.y, cv=kf)
                self.results_3.append(result)

        np_testing.assert_array_almost_equal(
            np.array(self.results_3).mean(axis=1), np.array(self.exercise.results_3).mean(axis=1), decimal=0)
        np_testing.assert_array_almost_equal(
            np.array(self.results_3).std(axis=1), np.array(self.exercise.results_3).std(axis=1), decimal=0)

if __name__ == '__main__':
    unittest.main()
