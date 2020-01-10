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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def build_model_1(activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(4, input_dim=28, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_model_2(activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(4, input_dim=28, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_model_3(activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(8, input_dim=28, activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Activity4_02
        self.activity = Activity4_02
        
        dirname = self._dirname_if_file('../data/HCV_feats.csv')
        self.X_loc = os.path.join(dirname, 'HCV_feats.csv')
        self.y_loc = os.path.join(dirname, 'HCV_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        sc = StandardScaler()
        self.X = pd.DataFrame(sc.fit_transform(self.X), columns=self.X.columns)

        self.seed = 2
        self.n_folds = 5
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)

    def test_model_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        batch_size=20
        epochs=50

        self.results_1 =[]
        models = [build_model_1, build_model_2, build_model_3]
        for m in range(len(models)):
            classifier = KerasClassifier(build_fn=models[m], epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
            kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            result = cross_val_score(classifier, self.X, self.y, cv=kfold)
            self.results_1.append(result)
        
        np_testing.assert_array_almost_equal(
            np.array(self.results_1).mean(axis=1), np.array(self.activity.results_1).mean(axis=1), decimal=1)
        np_testing.assert_array_almost_equal(
            np.array(self.results_1).std(axis=1), np.array(self.activity.results_1).std(axis=1), decimal=1)

    def test_batch_epoch_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        n_folds = 5
        epochs = [100, 200]
        batches = [10, 20]
        self.results_2 =[]
        for e in range(len(epochs)):
            for b in range(len(batches)):
                classifier = KerasClassifier(build_fn=build_model_2, epochs=epochs[e], batch_size=batches[b], verbose=0, shuffle=False)
                kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                result = cross_val_score(classifier, self.X, self.y, cv=kfold)
                self.results_2.append(result)
        
        np_testing.assert_array_almost_equal(
            np.array(self.results_2).mean(axis=1), np.array(self.activity.results_2).mean(axis=1), decimal=1)
        np_testing.assert_array_almost_equal(
            np.array(self.results_2).std(axis=1), np.array(self.activity.results_2).std(axis=1), decimal=1)

    def test_opt_act_iter(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        n_folds = 5
        batch_size=20
        epochs=100
        self.results_3 =[]
        optimizers = ['rmsprop', 'adam','sgd']
        activations = ['relu', 'tanh']
        for o in range(len(optimizers)):
            for a in range(len(activations)):
                optimizer = optimizers[o]
                activation = activations[a]
                classifier = KerasClassifier(build_fn=build_model_2, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
                kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                result = cross_val_score(classifier, self.X, self.y, cv=kfold)
                self.results_3.append(result)

        np_testing.assert_array_almost_equal(
            np.array(self.results_3).mean(axis=1), np.array(self.activity.results_3).mean(axis=1), decimal=1)
        np_testing.assert_array_almost_equal(
            np.array(self.results_3).std(axis=1), np.array(self.activity.results_3).std(axis=1), decimal=1)

if __name__ == '__main__':
    unittest.main()
