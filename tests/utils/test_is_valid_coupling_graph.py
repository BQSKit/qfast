import numpy    as np
import unittest as ut

from qfast.utils import is_valid_coupling_graph


class TestIsValidCouplingGraph ( ut.TestCase ):
    
    def test_is_valid_coupling_graph_empty ( self ):
        self.assertTrue( is_valid_coupling_graph( [] ) )
        self.assertTrue( is_valid_coupling_graph( [], 0 ) )
        self.assertTrue( is_valid_coupling_graph( [], 5 ) )

    def test_is_valid_coupling_graph_valid ( self ):
        cgraph = [ (0, 1), (1, 2), (2, 3), (4, 3), (0, 4) ]
        self.assertTrue( is_valid_coupling_graph( cgraph ) )
        self.assertTrue( is_valid_coupling_graph( cgraph, 5 ) )

    def test_is_valid_coupling_graph_invalid1 ( self ):
        self.assertRaises( TypeError, is_valid_coupling_graph, [(0,1)], "a" )
    
    def test_is_valid_coupling_graph_invalid2 ( self ):
        self.assertFalse( is_valid_coupling_graph( "a" ) )
        self.assertFalse( is_valid_coupling_graph( "a", 0 ) )

    def test_is_valid_coupling_graph_invalid3 ( self ):
        self.assertFalse( is_valid_coupling_graph( [ "a" ] ) )
        self.assertFalse( is_valid_coupling_graph( [ [0, 1] ] ) )
        self.assertFalse( is_valid_coupling_graph( [ [ [ 0 ], 1] ] ) )

        self.assertFalse( is_valid_coupling_graph( [ "a" ], 2 ) )
        self.assertFalse( is_valid_coupling_graph( [ [0, 1] ], 2 ) )
        self.assertFalse( is_valid_coupling_graph( [ [ [ 0 ], 1] ], 2 ) )

    def test_is_valid_coupling_graph_invalid4 ( self ):
        self.assertFalse( is_valid_coupling_graph( [ (0, 1, 2) ] ) )
        self.assertFalse( is_valid_coupling_graph( [ (0, 1, 2) ], 4 ) )
    
    def test_is_valid_coupling_graph_invalid5 ( self ):
        self.assertFalse( is_valid_coupling_graph( [ (0, 1), (1, 2) ], 1 ) )

    def test_is_valid_coupling_graph_invalid6 ( self ):
        self.assertFalse( is_valid_coupling_graph( [ (0, 1), (0, 1) ] ) )
        self.assertFalse( is_valid_coupling_graph( [ (0, 1), (0, 1) ], 5 ) )

    def test_is_valid_coupling_graph_invalid7 ( self ):
        self.assertFalse( is_valid_coupling_graph( [ (0, 0), (1, 1) ] ) )
        self.assertFalse( is_valid_coupling_graph( [ (0, 0), (1, 1) ], 5 ) )

if __name__ == '__main__':
    ut.main()
