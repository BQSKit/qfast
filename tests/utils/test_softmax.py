import numpy    as np
import unittest as ut

from qfast.utils import softmax


class TestSoftmax ( ut.TestCase ):
    
    def test_softmax1 ( self ):
        for i in range( 10 ):
            x = 10 * np.random.random( 100 )
            self.assertTrue( np.abs( np.sum( softmax( x ) ) - 1 ) < 1e-15 )

    def test_softmax2 ( self ):
        x = np.ones( 10 )
        x[0] = 2
        self.assertTrue( np.argmax( softmax( x ) ) == 0 )
        x[0] = 1
        x[5] = 2
        self.assertTrue( np.argmax( softmax( x ) ) == 5 )

    def test_softmax_invalid ( self ):
        self.assertRaises( TypeError, softmax, "a" )
    

if __name__ == '__main__':
    ut.main()
