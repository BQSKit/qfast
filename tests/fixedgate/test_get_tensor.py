import tensorflow as tf
import numpy      as np
import scipy.linalg as la

from qfast import hilbert_schmidt_distance
from qfast import FixedGate, get_pauli_n_qubit_projection
from qfast import pauli_dot_product, reset_tensor_cache


class TestFixedgateGetTensor ( tf.test.TestCase ):

    def test_fixedgate_get_tensor ( self ):
        reset_tensor_cache()
        fg = FixedGate( "Test", 4, 2, (0, 1) )
        tensor = fg.get_tensor()

        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer() )
            tensor = tensor.eval()

        paulis = get_pauli_n_qubit_projection( 4, (0, 1) )
        H = pauli_dot_product( [ 0.25 ] * 16, paulis )
        U = la.expm( 1j * H )
        self.assertTrue( hilbert_schmidt_distance( tensor, U ) <= 1e-15 )


if __name__ == '__main__':
    tf.test.main()
