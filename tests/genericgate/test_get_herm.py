import tensorflow as tf
import numpy      as np

from qfast import LocationModel
from qfast import GenericGate, get_pauli_n_qubit_projection
from qfast import pauli_dot_product, reset_tensor_cache


class TestGenericgateGetHerm ( tf.test.TestCase ):

    def test_genericgate_get_herm ( self ):
        reset_tensor_cache()
        lm = LocationModel( 4, 2 )
        gg = GenericGate( "Test", 4, 2, lm, loc_vals = [ 1, 0, 0, 0, 0, 0 ] )
        herm = gg.get_herm()

        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer() )
            herm = herm.eval()

        paulis = get_pauli_n_qubit_projection( 4, list( lm.locations )[0] )
        H = pauli_dot_product( [ 0.25 ] * 16, paulis )
        self.assertTrue( np.allclose( herm, H, rtol = 0, atol = 1e-15 )  )


if __name__ == '__main__':
    tf.test.main()
