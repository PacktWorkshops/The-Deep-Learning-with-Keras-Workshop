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
from sklearn.preprocessing import StandardScaler

        
class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))
     
    def setUp(self):
        import Activity3_02
        self.activity = Activity3_02
        
        dirname = self._dirname_if_file('../data/HCV_feats.csv')
        self.X_loc = os.path.join(dirname, 'HCV_feats.csv')
        self.y_loc = os.path.join(dirname, 'HCV_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)

        seed=1
        np.random.seed(seed)
        random.set_seed(seed)
        sc = StandardScaler()
        self.X = pd.DataFrame(sc.fit_transform(self.X), columns=self.X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed)
        
        np.random.seed(seed)
        random.set_seed(seed)
        classifier = Sequential()
        classifier.add(Dense(units = 4, activation = 'tanh', input_dim = self.X_train.shape[1]))
        classifier.add(Dense(units = 2, activation = 'tanh'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100, validation_split=0.1, shuffle=False)
        self.test_loss, self.test_acc = classifier.evaluate(self.X_test, self.y_test['AdvancedFibrosis'])
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)
        
    def test_model_perf(self):
        np_testing.assert_almost_equal(self.activity.test_loss, self.test_loss, decimal=2)
        np_testing.assert_almost_equal(self.activity.test_acc, self.test_acc, decimal=1)

if __name__ == '__main__':
    unittest.main()
