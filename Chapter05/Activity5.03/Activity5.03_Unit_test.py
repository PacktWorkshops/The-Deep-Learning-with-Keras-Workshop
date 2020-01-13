import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
from tensorflow import random
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_model(lambda_parameter):
    model = Sequential()
    model.add(Dense(10, input_dim=X.shape[1], activation='relu', kernel_regularizer=l2(lambda_parameter)))
    model.add(Dense(6, activation='relu', kernel_regularizer=l2(lambda_parameter)))
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(lambda_parameter)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Activity5_03
        self.activity = Activity5_03
        
        dirname = self._dirname_if_file('../data/avila-tr_feats.csv')
        self.X_loc = os.path.join(dirname, 'avila-tr_feats.csv')
        self.y_loc = os.path.join(dirname, 'avila-tr_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        self.seed = 1
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)

    def test_param_set_1(self):
        def build_model(lambda_parameter):
            model = Sequential()
            model.add(Dense(10, input_dim=self.X.shape[1], activation='relu', kernel_regularizer=l2(lambda_parameter)))
            model.add(Dense(6, activation='relu', kernel_regularizer=l2(lambda_parameter)))
            model.add(Dense(4, activation='relu', kernel_regularizer=l2(lambda_parameter)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            return model
        
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        model = KerasClassifier(build_fn=build_model, verbose=0, shuffle=False)
        lambda_parameter = [0.01, 0.5, 1]
        epochs = [50, 100]
        batch_size = [20]
        param_grid = dict(lambda_parameter=lambda_parameter, epochs=epochs, batch_size=batch_size)
        grid_seach = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        self.results_1 = grid_seach.fit(self.X, self.y)
        
        np_testing.assert_approx_equal(self.activity.results_1.best_score_,
                                       self.results_1.best_score_, significant=1)

        
    def test_param_set_2(self):
        def build_model(lambda_parameter):
            model = Sequential()
            model.add(Dense(10, input_dim=self.X.shape[1], activation='relu',
                            kernel_regularizer=l2(lambda_parameter)))
            model.add(Dense(6, activation='relu', kernel_regularizer=l2(lambda_parameter)))
            model.add(Dense(4, activation='relu', kernel_regularizer=l2(lambda_parameter)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            return model
        
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        model = KerasClassifier(build_fn=build_model, verbose=0, shuffle=False)
        lambda_parameter = [0.001, 0.01, 0.05, 0.1]
        epochs = [100]
        batch_size = [20]
        param_grid = dict(lambda_parameter=lambda_parameter, epochs=epochs, batch_size=batch_size)
        grid_seach = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        self.results_2 = grid_seach.fit(self.X, self.y)
        
        np_testing.assert_approx_equal(self.activity.results_2.best_score_,
                                       self.results_2.best_score_, significant=1)

    def test_param_set_3(self):
        def build_model(rate):
            model = Sequential()
            model.add(Dense(10, input_dim=self.X.shape[1], activation='relu'))
            model.add(Dropout(rate))
            model.add(Dense(6, activation='relu'))
            model.add(Dropout(rate))
            model.add(Dense(4, activation='relu'))
            model.add(Dropout(rate))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            return model
        
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        model = KerasClassifier(build_fn=build_model, verbose=0, shuffle=False)
        rate = [0, 0.1, 0.2]
        epochs = [50, 100]
        batch_size = [20]
        param_grid = dict(rate=rate, epochs=epochs, batch_size=batch_size)
        grid_seach = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        self.results_3 = grid_seach.fit(self.X, self.y)
        
        np_testing.assert_approx_equal(self.activity.results_3.best_score_,
                                       self.results_3.best_score_, significant=1)

    def test_param_set_4(self):
        def build_model(rate):
            model = Sequential()
            model.add(Dense(10, input_dim=self.X.shape[1], activation='relu'))
            model.add(Dropout(rate))
            model.add(Dense(6, activation='relu'))
            model.add(Dropout(rate))
            model.add(Dense(4, activation='relu'))
            model.add(Dropout(rate))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            return model
        
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        model = KerasClassifier(build_fn=build_model, verbose=0, shuffle=False)
        rate = [0.0, 0.05, 0.1]
        epochs = [100]
        batch_size = [20]
        param_grid = dict(rate=rate, epochs=epochs, batch_size=batch_size)
        grid_seach = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        self.results_4 = grid_seach.fit(self.X, self.y)
        
        np_testing.assert_approx_equal(self.activity.results_4.best_score_,
                                       self.results_4.best_score_, significant=1)
        
if __name__ == '__main__':
    unittest.main()
