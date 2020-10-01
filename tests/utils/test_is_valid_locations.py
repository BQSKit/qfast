import numpy    as np
import unittest as ut

from qfast.utils import is_valid_locations


class TestIsValidLocations ( ut.TestCase ):

    def test_is_valid_locations_empty ( self ):
        self.assertTrue( is_valid_locations( [] ) )
        self.assertTrue( is_valid_locations( [], 0 ) )
        self.assertTrue( is_valid_locations( [], 0, 0 ) )
        self.assertTrue( is_valid_locations( [], 256 ) )
        self.assertTrue( is_valid_locations( [], 256, 256 ) )

    def test_is_valid_locations_valid ( self ):
        locations = [ (0, 1, 2), (1, 2, 3), (3, 4, 5) ]
        self.assertTrue( is_valid_locations( locations ) )
        self.assertTrue( is_valid_locations( locations, 6 ) )
        self.assertTrue( is_valid_locations( locations, 6, 3 ) )
    
    def test_is_valid_locations_invalid1 ( self ):
        self.assertFalse( is_valid_locations( "a" ) )
        self.assertFalse( is_valid_locations( "a" , 0 ) )
        self.assertFalse( is_valid_locations( "a" , 0, 0 ) )
        self.assertFalse( is_valid_locations( "a" , 256 ) )
        self.assertFalse( is_valid_locations( "a" , 256, 256 ) )

    def test_is_valid_locations_invalid2 ( self ):
        self.assertFalse( is_valid_locations( [ "a" ] ) )
        self.assertFalse( is_valid_locations( [ "a" ], 0 ) )
        self.assertFalse( is_valid_locations( [ "a" ], 0, 0 ) )
        self.assertFalse( is_valid_locations( [ "a" ], 256 ) )
        self.assertFalse( is_valid_locations( [ "a" ], 256, 256 ) )

    def test_is_valid_locations_invalid3 ( self ):
        self.assertFalse( is_valid_locations( [ ( "a" ) ] ) )
        self.assertFalse( is_valid_locations( [ ( "a" ) ], 0 ) )
        self.assertFalse( is_valid_locations( [ ( "a" ) ], 0, 0 ) )
        self.assertFalse( is_valid_locations( [ ( "a" ) ], 256 ) )
        self.assertFalse( is_valid_locations( [ ( "a" ) ], 256, 256 ) )

    def test_is_valid_locations_invalid4 ( self ):
        self.assertFalse( is_valid_locations( [ (0, 1), (3) ] ) )
        self.assertFalse( is_valid_locations( [ (0, 1), (3) ], 4, 1 ) )
        self.assertFalse( is_valid_locations( [ (0, 1), (3) ], 4, 5 ) )

    def test_is_valid_locations_invalid5 ( self ):
        self.assertFalse( is_valid_locations( [ (0, 1), (0, 1) ] ) )
        self.assertFalse( is_valid_locations( [ (0, 1), (0, 1) ], 2 ) )
        self.assertFalse( is_valid_locations( [ (0, 1), (0, 1) ], 2, 2 ) )
    

if __name__ == '__main__':
    ut.main()

