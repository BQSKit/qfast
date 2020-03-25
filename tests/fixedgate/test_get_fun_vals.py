import tensorflow as tf
import numpy      as np
import scipy.linalg as la

from qfast import hilbert_schmidt_distance
from qfast import FixedGate, get_pauli_n_qubit_projection
from qfast import pauli_dot_product, reset_tensor_cache


class TestFixedgateGetFunVals ( tf.test.TestCase ):

    def test_fixedgate_get_fun_vals ( self ):
        reset_tensor_cache()
        fg = FixedGate( "Test", 4, 2, (0, 1) )

        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer() )
            fun_vals = fg.get_fun_vals( sess )

        self.assertTrue( np.array_equal( fun_vals, [ 0.25 ] * 16 ) )


if __name__ == '__main__':
    tf.test.main()
