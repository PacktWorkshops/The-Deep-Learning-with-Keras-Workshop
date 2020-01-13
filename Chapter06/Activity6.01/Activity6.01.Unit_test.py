import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import random

class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Activity6_01
        self.activity = Activity6_01
        
        dirname = self._dirname_if_file('../data/aps_failure_training_feats.csv')
        self.X_loc = os.path.join(dirname, 'aps_failure_training_feats.csv')
        self.y_loc = os.path.join(dirname, 'aps_failure_training_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        self.seed = 13
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.30, random_state=self.seed)
        
        sc=StandardScaler()

        self.X_train = sc.fit_transform(self.X_train)
        self.X_train = pd.DataFrame(self.X_train, columns=self.X_test.columns)

        self.X_test=sc.transform(self.X_test)
        self.X_test=pd.DataFrame(self.X_test, columns=self.X_train.columns)
        
    def test_input_frame(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)

    def test_model_perf(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        model=Sequential()
        model.add(Dense(units=64, activation='relu', kernel_initializer='uniform', input_dim=self.X_train.shape[1]))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=32, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(rate=0.4))
        model.add(Dense(units=16, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=8, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=4, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=100, batch_size=20, verbose=0, validation_split=0.2, shuffle=False)
        self.test_loss, self.test_acc = model.evaluate(self.X_test, self.y_test)
        
        np_testing.assert_approx_equal(self.activity.test_loss,
                                       self.test_loss, significant=0)
        
    def test_null_acc(self):
        ex_null = self.activity.y_test['class'].value_counts(normalize=True).loc[0]
        null = self.y_test['class'].value_counts(normalize=True).loc[0]
        np_testing.assert_equal(ex_null, null)
if __name__ == '__main__':
    unittest.main()
