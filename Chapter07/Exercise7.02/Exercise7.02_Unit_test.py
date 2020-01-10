import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

class Test(unittest.TestCase):
    
    def setUp(self):
        import Exercise7_02
        self.exercise = Exercise7_02
        self.seed = 42
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        
        self.classifier=Sequential()
        self.classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPool2D(pool_size = (2, 2)))
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPool2D(pool_size = (2, 2)))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units = 128, activation = 'relu'))
        self.classifier.add(Dense(128,activation='relu'))
        self.classifier.add(Dense(128,activation='relu'))
        self.classifier.add(Dense(128,activation='relu'))
        self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

        test_set = test_datagen.flow_from_directory('../dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
        self.classifier.fit_generator(
            training_set, steps_per_epoch = 10000, epochs = 2,
            validation_data = test_set, validation_steps = 2500,
            shuffle=False)
        
    def test_model_perf(self):
        np_testing.assert_approx_equal(self.exercise.classifier.history.history['val_accuracy'][0],
                                       self.classifier.history.history['val_accuracy'][0], significant=1)



if __name__ == '__main__':
    unittest.main()
