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
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1_l2

class Test(unittest.TestCase):
    
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))


    def setUp(self):
        import Activity5_01
        self.activity = Activity5_01
        
        dirname = self._dirname_if_file('../data/avila-tr_feats.csv')
        self.X_loc = os.path.join(dirname, 'avila-tr_feats.csv')
        self.y_loc = os.path.join(dirname, 'avila-tr_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)
        
        self.seed = 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seed)
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)

    def test_model_1(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        model_1 = Sequential()
        model_1.add(Dense(10, input_dim=self.X_test.shape[1], activation='relu'))
        model_1.add(Dense(6, activation='relu'))
        model_1.add(Dense(4, activation='relu'))
        model_1.add(Dense(1, activation='sigmoid'))
        model_1.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model_1.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100, validation_data=(self.X_test, self.y_test), verbose=0)
        loss, acc = model_1.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_1.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)

        
    def test_model_2(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        l2_param = 0.01
        model_2 = Sequential()
        model_2.add(Dense(10, input_dim=self.X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_param)))
        model_2.add(Dense(6, activation='relu', kernel_regularizer=l2(l2_param)))
        model_2.add(Dense(4, activation='relu', kernel_regularizer=l2(l2_param)))
        model_2.add(Dense(1, activation='sigmoid'))
        model_2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        model_2.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100, validation_data=(self.X_test, self.y_test), verbose=0)

        loss, acc = model_2.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_2.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)

    def test_model_3(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        l2_param = 0.1
        model_3 = Sequential()
        model_3.add(Dense(10, input_dim=self.X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_param)))
        model_3.add(Dense(6, activation='relu', kernel_regularizer=l2(l2_param)))
        model_3.add(Dense(4, activation='relu', kernel_regularizer=l2(l2_param)))
        model_3.add(Dense(1, activation='sigmoid'))
        model_3.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # train the model using training set while evaluationg on test set
        model_3.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100,
                    validation_data=(self.X_test, self.y_test), verbose=0)
        loss, acc = model_3.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_3.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)
        
    def test_model_4(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        l2_param = 0.005
        model_4 = Sequential()
        model_4.add(Dense(10, input_dim=self.X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_param)))
        model_4.add(Dense(6, activation='relu', kernel_regularizer=l2(l2_param)))
        model_4.add(Dense(4, activation='relu', kernel_regularizer=l2(l2_param)))
        model_4.add(Dense(1, activation='sigmoid'))
        model_4.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        model_4.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100,
                    validation_data=(self.X_test, self.y_test), verbose=0)

        loss, acc = model_4.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_4.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)

    def test_model_5(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        l1_param = 0.01
        model_5 = Sequential()
        model_5.add(Dense(10, input_dim=self.X_train.shape[1], activation='relu', kernel_regularizer=l1(l1_param)))
        model_5.add(Dense(6, activation='relu', kernel_regularizer=l1(l1_param)))
        model_5.add(Dense(4, activation='relu', kernel_regularizer=l1(l1_param)))
        model_5.add(Dense(1, activation='sigmoid'))
        model_5.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # train the model using training set while evaluationg on test set
        history=model_5.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100,
                            validation_data=(self.X_test, self.y_test), verbose=0)


        loss, acc = model_5.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_5.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)

    def test_model_6(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        l1_param = 0.1
        model_6 = Sequential()
        model_6.add(Dense(10, input_dim=self.X_train.shape[1], activation='relu', kernel_regularizer=l1(l1_param)))
        model_6.add(Dense(6, activation='relu', kernel_regularizer=l1(l1_param)))
        model_6.add(Dense(4, activation='relu', kernel_regularizer=l1(l1_param)))
        model_6.add(Dense(1, activation='sigmoid'))
        model_6.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        model_6.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100,
                    validation_data=(self.X_test, self.y_test), verbose=0)

        loss, acc = model_6.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_6.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)


    def test_model_7(self):
        np.random.seed(self.seed)
        random.set_seed(self.seed)

        l1_param = 0.01
        l2_param = 0.005
        model_7 = Sequential()
        model_7.add(Dense(10, input_dim=self.X_train.shape[1], activation='relu',
                          kernel_regularizer=l1_l2(l1=l1_param, l2=l2_param)))
        model_7.add(Dense(6, activation='relu', kernel_regularizer=l1_l2(l1=l1_param, l2=l2_param)))
        model_7.add(Dense(4, activation='relu', kernel_regularizer=l1_l2(l1=l1_param, l2=l2_param)))
        model_7.add(Dense(1, activation='sigmoid'))
        model_7.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model_7.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100,
                    validation_data=(self.X_test, self.y_test), verbose=0)
        
        loss, acc = model_7.evaluate(self.X_test, self.y_test)
        act_loss, act_acc = self.activity.model_7.evaluate(self.activity.X_test, self.activity.y_test)
        np_testing.assert_approx_equal(loss, act_loss, significant=2)

if __name__ == '__main__':
    unittest.main()
