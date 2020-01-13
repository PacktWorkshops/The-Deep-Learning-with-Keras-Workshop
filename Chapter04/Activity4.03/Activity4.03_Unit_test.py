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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model_1(optimizer='adam'):
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


# Create the function that returns the keras model 2
def build_model_2(optimizer='adam'):
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create the function that returns the keras model 3
def build_model_3(optimizer='adam'):
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
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
        import Activity4_03
        self.activity = Activity4_03
        
        dirname = self._dirname_if_file('../data/traffic_volume_feats.csv')
        self.X_loc = os.path.join(dirname, 'traffic_volume_feats.csv')
        self.y_loc = os.path.join(dirname, 'traffic_volume_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        self.seed=1
        self.n_folds = 5
        
            
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)

    def test_model_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        self.results_1 = []
        models = [build_model_1, build_model_2, build_model_3]
        for i in range(len(models)):
            regressor = KerasRegressor(build_fn=models[i], epochs=100, batch_size=50, verbose=0)
            model = make_pipeline(StandardScaler(), regressor)
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            result = cross_val_score(model, self.X, self.y, cv=kfold)
            self.results_1.append(result)

        np_testing.assert_almost_equal(
            np.array(self.results_1).mean(axis=1), np.array(self.activity.results_1).mean(axis=1), decimal=0)
        np_testing.assert_almost_equal(
            np.array(self.results_1).std(axis=1), np.array(self.activity.results_1).std(axis=1), decimal=0)

    def test_batch_epoch_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        self.results_2 = []
        epochs = [80, 100]
        batches = [50, 25]
        for i in range(len(epochs)):
            for j in range(len(batches)):
                regressor = KerasRegressor(build_fn=build_model_2, epochs=epochs[i], batch_size=batches[j], verbose=0)
                model = make_pipeline(StandardScaler(), regressor)
                kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                result = cross_val_score(model, self.X, self.y, cv=kfold)
                self.results_2.append(result)

        np_testing.assert_almost_equal(
            np.array(self.results_2).mean(axis=1), np.array(self.activity.results_2).mean(axis=1), decimal=0)
        np_testing.assert_almost_equal(
            np.array(self.results_2).std(axis=1), np.array(self.activity.results_2).std(axis=1), decimal=0)

    def test_opt_act_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        self.results_3 = []
        optimizers = ['adam', 'sgd', 'rmsprop']
        # loop over optimizers
        for optimizer in optimizers:
            regressor = KerasRegressor(build_fn=build_model_2, epochs=100, batch_size=50, verbose=0)
            model = make_pipeline(StandardScaler(), regressor)
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            result = cross_val_score(model, self.X, self.y, cv=kfold)
            self.results_3.append(result)

        np_testing.assert_almost_equal(
            np.array(self.results_3).mean(axis=1), np.array(self.activity.results_3).mean(axis=1), decimal=0)
        np_testing.assert_almost_equal(
            np.array(self.results_3).std(axis=1), np.array(self.activity.results_3).std(axis=1), decimal=0)

if __name__ == '__main__':
    unittest.main()
