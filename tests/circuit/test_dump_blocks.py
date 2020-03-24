import os
import tensorflow as tf
import numpy      as np

from unittest.mock import mock_open, patch

from qfast import Block, Circuit


class TestCircuitDumpBlocks ( tf.test.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_circuit_dump_blocks_invalid ( self ):
        circ = Circuit( self.TOFFOLI )
        circ.blocks = [ Block( self.TOFFOLI, (1, 2, 3) ),
                        Block( self.TOFFOLI, (0, 1, 2) ) ]
        self.assertRaises( ValueError, circ.dump_blocks, ":+?\\=-+./*!" )

    def test_circuit_dump_blocks_valid ( self ):
        circ = Circuit( self.TOFFOLI )
        circ.blocks = [ Block( self.TOFFOLI, (1, 2, 3) ),
                        Block( self.TOFFOLI, (0, 1, 2) ) ]

        circ.dump_blocks( "." )
        self.assertTrue( os.path.isfile( "./0_1_2_3.unitary" ) )
        self.assertTrue( os.path.isfile( "./1_0_1_2.unitary" ) )
        os.remove( "./0_1_2_3.unitary" )
        os.remove( "./1_0_1_2.unitary" )
