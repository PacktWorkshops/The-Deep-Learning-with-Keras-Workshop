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
        import Exercise3_01
        self.exercise = Exercise3_01
        
        dirname = self._dirname_if_file('../data/tree_class_feats.csv')
        self.X_loc = os.path.join(dirname, 'tree_class_feats.csv')
        self.y_loc = os.path.join(dirname, 'tree_class_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)

        np.random.seed(1)
        random.set_seed(1)
        model = Sequential()
        model.add(Dense(10, activation='tanh', input_dim=self.X.shape[1]))
        model.add(Dense(5, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.X, self.y, epochs=100, batch_size=5, verbose=0, validation_split=0.2, shuffle=False)
        self.y_predicted = model.predict(self.X.iloc[0:10,:])
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.exercise.X, self.X)
        pd_testing.assert_frame_equal(self.exercise.y, self.y)

    def test_model_perf(self):
        np_testing.assert_array_equal(np.round(self.exercise.y_predicted), np.round(self.y_predicted))

if __name__ == '__main__':
    unittest.main()
