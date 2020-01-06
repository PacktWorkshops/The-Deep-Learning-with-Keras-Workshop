import unittest
import import_ipynb
import numpy as np
import numpy.testing as np_testing
import os

class Test(unittest.TestCase):
     
    def setUp(self):
        import Exercise2_02
        self.exercise = Exercise2_02
        
        self.mat1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.mat2 = np.reshape(self.mat1, [3,4])
        self.mat3 = np.reshape(self.mat1, [3,2,2])
        self.mat4 = np.reshape(self.mat1, [12])
      
    def test_matrix_reshaping1(self):
        np_testing.assert_equal(self.exercise.mat2, self.mat2)

    def test_matrix_reshaping2(self):
        np_testing.assert_equal(self.exercise.mat3, self.mat3)

    def test_matrix_reshaping3(self):
        np_testing.assert_equal(self.exercise.mat4, self.mat4)


if __name__ == '__main__':
    unittest.main()
