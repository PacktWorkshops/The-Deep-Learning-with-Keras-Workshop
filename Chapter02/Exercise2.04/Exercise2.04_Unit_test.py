import unittest
import import_ipynb
import numpy as np
import numpy.testing as np_testing
import os

class Test(unittest.TestCase):
     
    def setUp(self):
        import Exercise2_04
        self.exercise = Exercise2_04
        
        self.mat1 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        self.mat2 = self.mat1.T
        self.mat3 = self.mat2.dot(self.mat1)
      
    def test_matrix_transposition(self):
        np_testing.assert_equal(self.exercise.mat2, self.mat2)

    def test_matrix_multiplication(self):
        np_testing.assert_equal(self.exercise.mat3, self.mat3)


if __name__ == '__main__':
    unittest.main()
