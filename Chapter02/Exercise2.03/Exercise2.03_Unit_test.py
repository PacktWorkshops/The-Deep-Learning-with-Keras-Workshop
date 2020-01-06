import unittest
import import_ipynb
import numpy as np
import numpy.testing as np_testing
import os

class Test(unittest.TestCase):
     
    def setUp(self):
        import Exercise2_03
        self.exercise = Exercise2_03
        
        self.mat1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.mat2 = np.array([[2, 1, 4], [4, 1, 7], [4, 2, 9], [5, 21, 1]])
        self.mat3 = self.mat1.dot(self.mat2.T)
        self.mat4 = self.mat1.T.dot(self.mat2)
        self.mat5 = np.reshape(self.mat1, [3,4]).dot(self.mat2)
      
    def test_matrix_multiplication1(self):
        ex2_03_mat3 = np.array([[ 16,  27,  35,  50], [ 37,  63,  80, 131],
                             [ 58,  99, 125, 212], [ 79, 135, 170, 293]])
        np_testing.assert_equal(ex2_03_mat3, self.mat3)

    def test_matrix_multiplication2(self):
        ex2_03_mat4 = np.array([[ 96, 229, 105], [111, 254, 126], [126, 279, 147]])
        np_testing.assert_equal(ex2_03_mat4, self.mat4)

    def test_matrix_multiplication3(self):
        ex2_03_mat5 = np.array([[ 42,  93,  49], [102, 193, 133], [162, 293, 217]])
        np_testing.assert_equal(ex2_03_mat5, self.mat5)

if __name__ == '__main__':
    unittest.main()
