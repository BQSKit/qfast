import numpy    as np
import unittest as ut

from qfast.utils import is_valid_location


class TestIsValidLocation ( ut.TestCase ):
    
    def test_is_valid_location_invalid1 ( self ):
        self.assertFalse( is_valid_location( "a" ) )
        self.assertFalse( is_valid_location( 0 ) )
        self.assertFalse( is_valid_location( "a", 256 ) )
        self.assertFalse( is_valid_location( 0, 256 ) )

    def test_is_valid_location_invalid2 ( self ):
        self.assertFalse( is_valid_location( (0, "a", 2) ) )
        self.assertFalse( is_valid_location( (0, "a", 2), 256 ) )

    def test_is_valid_location_invalid3 ( self ):
        self.assertFalse( is_valid_location( (0, 0) ) )
        self.assertFalse( is_valid_location( (0, 0), 256 ) )

    def test_is_valid_location_invalid4 ( self ):
        self.assertFalse( is_valid_location( (0, 1, 2), 1 ) )
        self.assertFalse( is_valid_location( (0, 1, 2), 0 ) )

    def test_is_valid_location_empty ( self ):
        self.assertTrue( is_valid_location( tuple() ) )
        self.assertTrue( is_valid_location( tuple(), 0 ) )
        self.assertTrue( is_valid_location( tuple(), 3 ) )
    
    def test_is_valid_location_valid ( self ):
        self.assertTrue( is_valid_location( (0, 1, 2) ) )
        self.assertTrue( is_valid_location( (0, 1, 2), 3 ) )
        self.assertTrue( is_valid_location( (0, 1, 2, 17) ) )
        self.assertTrue( is_valid_location( (0, 1, 2, 17), 18 ) )
    

if __name__ == '__main__':
    ut.main()
