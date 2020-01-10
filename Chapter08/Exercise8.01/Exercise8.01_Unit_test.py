import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import os
import import_ipynb
from tensorflow import random
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image

class Test(unittest.TestCase):
    
    def setUp(self):
        import Exercise8_01
        self.exercise = Exercise8_01
        
        self.classifier=VGG16()
        
        self.new_image = image.load_img('../Data/Prediction/pizza.jpg', target_size=(224, 224))
        self.transformed_image = image.img_to_array(self.new_image)
        self.transformed_image = np.expand_dims(self.transformed_image, axis=0)
        self.transformed_image = preprocess_input(self.transformed_image)
        
    def test_image_size(self):
        np_testing.assert_array_equal(self.transformed_image, self.exercise.transformed_image)

    def test_prediction(self):
        y_pred = self.classifier.predict(self.transformed_image)
        np_testing.assert_array_equal(y_pred, self.exercise.y_pred)

    def test_decode_prediction(self):
        y_pred = self.classifier.predict(self.transformed_image)
        predictions = decode_predictions(y_pred, top=5)
        np_testing.assert_array_equal(predictions, decode_predictions(self.exercise.y_pred, top=5))

if __name__ == '__main__':
    unittest.main()
