import tensorflow as tf
import numpy      as np

from qfast import Block, Circuit, unitary_log_no_i, pauli_expansion


class TestCircuitGetFunLocVals ( tf.test.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_circuit_get_fun_loc_vals ( self ):
        circ = Circuit( self.TOFFOLI )
        circ.blocks = [ Block( self.TOFFOLI, (1, 2, 3) ),
                        Block( self.TOFFOLI, (0, 1, 2) ) ]
        fun_loc_vals = circ.get_fun_loc_vals()

        self.assertTrue( isinstance( fun_loc_vals, tuple ) )
        self.assertEqual( len( fun_loc_vals ), 2 )

        fun_vals = fun_loc_vals[0]
        loc_vals = fun_loc_vals[1]

        self.assertTrue( isinstance( fun_vals, list ) )
        self.assertTrue( isinstance( loc_vals, list ) )

        self.assertEqual( len( fun_vals ), 2 )
        self.assertEqual( len( loc_vals ), 2 )

        toffoli_vals = pauli_expansion( unitary_log_no_i( self.TOFFOLI ) )
        self.assertTrue( np.array_equal( fun_vals[0], toffoli_vals ) )
        self.assertTrue( np.array_equal( fun_vals[1], toffoli_vals ) )

        self.assertTrue( np.array_equal( loc_vals[0], (1, 2, 3) ) )
        self.assertTrue( np.array_equal( loc_vals[1], (0, 1, 2) ) )
