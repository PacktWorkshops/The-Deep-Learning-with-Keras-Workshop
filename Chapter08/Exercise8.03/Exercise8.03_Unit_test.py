import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
from tensorflow import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras

class Test(unittest.TestCase):

    def setUp(self):
        import Exercise8_03
        self.exercise = Exercise8_03
        
        vgg_model = keras.applications.vgg16.VGG16()
        
        self.seed = 42
        np.random.seed(self.seed)
        random.set_seed(self.seed)
        last_layer = str(vgg_model.layers[-1])

        self.classifier= Sequential()
        for layer in vgg_model.layers:
            if str(layer) != last_layer:
                self.classifier.add(layer)
                
        for layer in self.classifier.layers:
            layer.trainable=False        
        self.classifier.add(Dense(1, activation='sigmoid'))
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        generate_train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

        generate_test_data = ImageDataGenerator(rescale = 1./255)        
        
        training_dataset = generate_train_data.flow_from_directory('../Data/Dataset/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

        test_dataset = generate_test_data.flow_from_directory('../Data/Dataset/test_set',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

        
        
        self.classifier.fit_generator(
            training_dataset, steps_per_epoch = 100, epochs = 10,
            validation_data = test_dataset, validation_steps = 30,
            shuffle=False)
        
    def test_model_perf(self):
        np_testing.assert_approx_equal(self.exercise.classifier.history.history['val_accuracy'][0],
                                       self.classifier.history.history['val_accuracy'][0], significant=1)

    def test_model_pred(self):
        new_image = image.load_img('../Data/Prediction/test_image_2.jpg', target_size = (224, 224))
        new_image = image.img_to_array(new_image)
        new_image = np.expand_dims(new_image, axis = 0)
        result = self.classifier.predict(new_image)
        np_testing.assert_approx_equal(self.exercise.result[0][0], result[0][0], significant=1)


if __name__ == '__main__':
    unittest.main()
