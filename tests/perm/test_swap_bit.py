import numpy    as np
import unittest as ut

from qfast.perm import swap_bit


class TestSwapBit ( ut.TestCase ):
    
    def test_swap_bit1 ( self ):
        self.assertTrue( swap_bit( 0, 1, 0 ) == 0 )
        self.assertTrue( swap_bit( 0, 1, 1 ) == 2 )
        self.assertTrue( swap_bit( 0, 1, 2 ) == 1 )
        self.assertTrue( swap_bit( 0, 1, 3 ) == 3 )
        self.assertTrue( swap_bit( 0, 1, 4 ) == 4 )
        self.assertTrue( swap_bit( 0, 1, 5 ) == 6 )
        self.assertTrue( swap_bit( 0, 1, 6 ) == 5 )
        self.assertTrue( swap_bit( 0, 1, 7 ) == 7 )
        self.assertTrue( swap_bit( 0, 1, 8 ) == 8 )

    def test_swap_bit2 ( self ):
        self.assertTrue( swap_bit( 1, 2, 0 ) == 0 )
        self.assertTrue( swap_bit( 1, 2, 1 ) == 1 )
        self.assertTrue( swap_bit( 1, 2, 2 ) == 4 )
        self.assertTrue( swap_bit( 1, 2, 3 ) == 5 )
        self.assertTrue( swap_bit( 1, 2, 4 ) == 2 )
        self.assertTrue( swap_bit( 1, 2, 5 ) == 3 )
        self.assertTrue( swap_bit( 1, 2, 6 ) == 6 )
        self.assertTrue( swap_bit( 1, 2, 7 ) == 7 )
        self.assertTrue( swap_bit( 1, 2, 8 ) == 8 )
    
    def test_swap_bit3 ( self ):
        for i in range( 10 ):
            for j in range( 10 ):
                self.assertTrue( swap_bit( i, j, 2112 ) == swap_bit( j, i, 2112 ) )

    def test_swap_bit_invalid ( self ):
        self.assertRaises( TypeError, swap_bit, "a", 1, 0 )
        self.assertRaises( TypeError, swap_bit, 0, "a", 0 )
        self.assertRaises( TypeError, swap_bit, 0, 1, "a" )


if __name__ == '__main__':
    ut.main()
