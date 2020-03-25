import tensorflow as tf
import numpy      as np


from qfast import FixedGate, get_pauli_n_qubit_projection
from qfast import pauli_dot_product, reset_tensor_cache


class TestFixedgateGetHerm ( tf.test.TestCase ):

    def test_fixedgate_get_herm ( self ):
        reset_tensor_cache()
        fg = FixedGate( "Test", 4, 2, (0, 1) )
        herm = fg.get_herm()

        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer() )
            herm = herm.eval()

        paulis = get_pauli_n_qubit_projection( 4, (0, 1) )
        H = pauli_dot_product( [ 0.25 ] * 16, paulis )
        self.assertTrue( np.array_equal( herm, H )  )


if __name__ == '__main__':
    tf.test.main()
