import numpy    as np
import unittest as ut

from qfast.topology import Topology 


class TestTopologyConstructor ( ut.TestCase ):

    def test_topology_constructor_cgraph ( self ):
        cgraph = [ (0, 1), (1, 2), (2, 3) ]
        t = Topology( 4, cgraph )

        self.assertTrue( len( t.coupling_graph ) == 3 )
        for link in cgraph:
            self.assertTrue( link in t.coupling_graph )

    def test_topology_constructor_num_qubits ( self ):
        for n in [ 1, 2, 3, 4 ]:
            t = Topology( n )
            self.assertTrue( t.num_qubits == n )

    def test_topology_constructor_alltoall ( self ):
        t = Topology( 2 )
        self.assertTrue( len( t.coupling_graph ) == 1 )
        self.assertTrue( (0, 1) in t.coupling_graph )

        t = Topology( 3 )
        self.assertTrue( len( t.coupling_graph ) == 3 )
        self.assertTrue( (0, 1) in t.coupling_graph )
        self.assertTrue( (0, 2) in t.coupling_graph )
        self.assertTrue( (1, 2) in t.coupling_graph )

        t = Topology( 4 )
        self.assertTrue( len( t.coupling_graph ) == 6 )
        self.assertTrue( (0, 1) in t.coupling_graph )
        self.assertTrue( (0, 2) in t.coupling_graph )
        self.assertTrue( (0, 3) in t.coupling_graph )
        self.assertTrue( (1, 2) in t.coupling_graph )
        self.assertTrue( (1, 3) in t.coupling_graph )
        self.assertTrue( (2, 3) in t.coupling_graph )

    def test_topology_constructor_cgraph_invalid ( self ):
        cgraph = [ (0, 1), (1, 2), (2, 3) ]
        self.assertRaises( TypeError, Topology, 2, cgraph )
        self.assertRaises( TypeError, Topology, 2, (0, 1) )
        self.assertRaises( TypeError, Topology, 2, 0 )
        self.assertRaises( TypeError, Topology, 2, "a" )


if __name__ == '__main__':
    ut.main()
