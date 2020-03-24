import tensorflow as tf
import numpy      as np

from qfast import Block, Circuit


class TestCircuitGetLocations ( tf.test.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_circuit_get_locations ( self ):
        circ = Circuit( self.TOFFOLI )
        circ.blocks = [ Block( self.TOFFOLI, (1, 2, 3) ),
                        Block( self.TOFFOLI, (0, 1, 2) ) ]
        locations = circ.get_locations()

        self.assertTrue( isinstance( locations, list ) )
        self.assertEqual( len( locations ), 2 )
        self.assertTrue( np.array_equal( locations[0], (1, 2, 3) ) )
        self.assertTrue( np.array_equal( locations[1], (0, 1, 2) ) )
