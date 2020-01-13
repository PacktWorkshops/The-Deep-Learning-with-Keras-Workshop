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

class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Exercise5_01
        self.exercise = Exercise5_01
        
        dirname = self._dirname_if_file('../data/tree_class_feats.csv')
        self.X_loc = os.path.join(dirname, 'tree_class_feats.csv')
        self.y_loc = os.path.join(dirname, 'tree_class_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        self.seed = 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seed)
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.exercise.X, self.X)
        pd_testing.assert_frame_equal(self.exercise.y, self.y)

    def test_model_1(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        model_1 = Sequential()
        model_1.add(Dense(16, activation='relu', input_dim=self.X_train.shape[1]))
        model_1.add(Dense(12, activation='relu'))
        model_1.add(Dense(8, activation='relu'))
        model_1.add(Dense(4, activation='relu'))
        model_1.add(Dense(1, activation='sigmoid'))

        model_1.compile(optimizer='sgd', loss='binary_crossentropy')
        model_1.fit(self.X_train, self.y_train, epochs=300, batch_size=50, verbose=0)
        loss = model_1.evaluate(self.X_test, self.y_test)
        ex_loss = self.exercise.model_1.evaluate(self.exercise.X_test, self.exercise.y_test)
        np_testing.assert_approx_equal(loss, ex_loss, significant=0)

        
    def test_model_2(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        model_2 = Sequential()
        model_2.add(Dense(16, activation='relu', input_dim=self.X_train.shape[1]))
        model_2.add(Dropout(0.1))
        model_2.add(Dense(12, activation='relu'))
        model_2.add(Dense(8, activation='relu'))
        model_2.add(Dense(4, activation='relu'))
        model_2.add(Dense(1, activation='sigmoid'))

        # Choose the loss function to be binary cross entropy and the optimizer to be SGD for training the model
        model_2.compile(optimizer='sgd', loss='binary_crossentropy')
        # train the model
        model_2.fit(self.X_train, self.y_train, epochs=300, batch_size=50, verbose=0)
        loss = model_2.evaluate(self.X_test, self.y_test)
        ex_loss = self.exercise.model_2.evaluate(self.exercise.X_test, self.exercise.y_test)
        np_testing.assert_approx_equal(loss, ex_loss, significant=0)

    def test_model_3(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        model_3 = Sequential()
        model_3.add(Dense(16, activation='relu', input_dim=self.X_train.shape[1]))
        model_3.add(Dropout(0.2))
        model_3.add(Dense(12, activation='relu'))
        model_3.add(Dropout(0.1))
        model_3.add(Dense(8, activation='relu'))
        model_3.add(Dropout(0.1))
        model_3.add(Dense(4, activation='relu'))
        model_3.add(Dropout(0.1))
        model_3.add(Dense(1, activation='sigmoid'))

        # Choose the loss function to be binary cross entropy and the optimizer to be SGD for training the model
        model_3.compile(optimizer='sgd', loss='binary_crossentropy')
        # train the model
        model_3.fit(self.X_train, self.y_train, epochs=300, batch_size=50, verbose=0)
        loss = model_3.evaluate(self.X_test, self.y_test)
        ex_loss = self.exercise.model_3.evaluate(self.exercise.X_test, self.exercise.y_test)
        np_testing.assert_approx_equal(loss, ex_loss, significant=0)
        
if __name__ == '__main__':
    unittest.main()
