import numpy    as np
import unittest as ut

from qfast.topology import Topology 


class TestTopologyGetLocations ( ut.TestCase ):

    def test_topology_get_locations_1 ( self ):
        cgraph = [ (0, 1), (1, 2), (2, 3) ]
        t = Topology( 4, cgraph )
        l = t.get_locations( 2 )

        self.assertTrue( len( l ) == 3 )
        self.assertTrue( (0, 1) in l )
        self.assertTrue( (1, 2) in l )
        self.assertTrue( (2, 3) in l )

    def test_topology_get_locations_2 ( self ):
        cgraph = [ (0, 1), (1, 2), (2, 3) ]
        t = Topology( 4, cgraph )
        l = t.get_locations( 3 )

        self.assertTrue( len( l ) == 2 )
        self.assertTrue( (0, 1, 2) in l )
        self.assertTrue( (1, 2, 3) in l )

    def test_topology_get_locations_3 ( self ):
        t = Topology( 4 )
        l = t.get_locations( 3 )

        self.assertTrue( len( l ) == 4 )
        self.assertTrue( (0, 1, 2) in l )
        self.assertTrue( (0, 1, 3) in l )
        self.assertTrue( (0, 2, 3) in l )
        self.assertTrue( (1, 2, 3) in l )

    def test_topology_get_locations_invalid ( self ):
        cgraph = [ (0, 1), (1, 2), (2, 3) ]
        t = Topology( 4, cgraph )

        self.assertRaises( ValueError, t.get_locations, 5 )
        self.assertRaises( ValueError, t.get_locations, 0 )
        self.assertRaises( ValueError, t.get_locations, -2 )
        self.assertRaises( TypeError, t.get_locations, "a" )


if __name__ == '__main__':
    ut.main()
