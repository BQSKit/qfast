import numpy    as np
import unittest as ut

from qfast.perm import swap


class TestSwap ( ut.TestCase ):
    
    def test_swap ( self ):
        perm = swap( 0, 1, 2 ).list()
        self.assertTrue( perm[0] == 0 )
        self.assertTrue( perm[1] == 2 )
        self.assertTrue( perm[2] == 1 )
        self.assertTrue( perm[3] == 3 )
        self.assertTrue( len( perm ) == 4 )

    def test_swap_invalid1 ( self ):
        self.assertRaises( TypeError, swap, "a", 1, 0 )
        self.assertRaises( TypeError, swap, 0, "a", 0 )
        self.assertRaises( TypeError, swap, 0, 1, "a" )

    def test_swap_invalid2 ( self ):
        self.assertRaises( ValueError, swap, 0, 1, 0 )
        self.assertRaises( ValueError, swap, 1, 2, 1 )


if __name__ == '__main__':
    ut.main()
