import unittest
import import_ipynb
import numpy as np
import numpy.testing as np_testing
import os

class Test(unittest.TestCase):
     
    def setUp(self):
        import Exercise2_01
        self.exercise = Exercise2_01
        
        self.mat1 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.mat2 = np.matrix([[2, 1, 4], [4, 1, 7], [4, 2, 9], [5, 21, 1]])
        self.mat3 = self.mat1 + self.mat2
      
    def test_matrix_addition(self):
        np_testing.assert_equal(self.exercise.mat3, self.mat3)


if __name__ == '__main__':
    unittest.main()
