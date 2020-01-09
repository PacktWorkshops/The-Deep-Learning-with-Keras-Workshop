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


def build_model():
    model = Sequential()
    model.add(Dense(4, input_dim=28, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Activity4_01
        self.activity = Activity4_01
        
        dirname = self._dirname_if_file('../data/HCV_feats.csv')
        self.X_loc = os.path.join(dirname, 'HCV_feats.csv')
        self.y_loc = os.path.join(dirname, 'HCV_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        sc = StandardScaler()
        self.X = pd.DataFrame(sc.fit_transform(self.X), columns=self.X.columns)

        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)

    def test_model(self):
        self.seed = 1
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        classifier = KerasClassifier(build_fn=build_model, epochs=100, batch_size=20, verbose=0, shuffle=False)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        self.results = cross_val_score(classifier, self.X, self.y, cv=kfold)
        
        np_testing.assert_almost_equal(self.results.mean(), self.activity.results.mean(), decimal=1)
        np_testing.assert_almost_equal(self.results.std(), self.activity.results.std(), decimal=1)


if __name__ == '__main__':
    unittest.main()
