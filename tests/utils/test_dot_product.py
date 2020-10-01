import numpy    as np
import unittest as ut

from qfast.utils import dot_product


class TestDotProduct ( ut.TestCase ):
    
    def test_dot_product_valid1 ( self ):
        sigma = np.array( [ [ [ 0, 1 ], [ 1, 0 ] ], [ [ 1, 0 ], [ 0, 1 ] ] ] )
        alpha = [ 1, 1 ]

        expected = np.array( [ [ 1, 1 ], [ 1, 1 ] ] )
        self.assertTrue( np.allclose( dot_product( alpha, sigma ), expected ) )
    
    def test_dot_product_valid2 ( self ):
        sigma = np.array( [ [ [ 0, 1 ], [ 1, 0 ] ], [ [ 1, 0 ], [ 0, 1 ] ] ] )
        alpha = [ 0.5, 0.5 ]

        expected = 0.5 * np.array( [ [ 1, 1 ], [ 1, 1 ] ] )
        self.assertTrue( np.allclose( dot_product( alpha, sigma ), expected ) )
    
    def test_dot_product_invalid ( self ):
        sigma = np.array( [ [ [ 0, 1 ], [ 1, 0 ] ], [ [ 1, 0 ], [ 0, 1 ] ] ] )
        alpha = [ 1 ]

        self.assertRaises( ValueError, dot_product, alpha, sigma )

    def test_dot_product_invalid ( self ):
        sigma = "a"
        alpha = "b"

        self.assertRaises( TypeError, dot_product, alpha, sigma )
    

if __name__ == '__main__':
    ut.main()
