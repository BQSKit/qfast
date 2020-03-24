import tensorflow as tf
import numpy      as np

from qfast import pauli_dot_product, I, X, Y, Z, get_norder_paulis


class TestPauliDotProduct ( tf.test.TestCase ):

    def test_pauli_dot_product_invalid ( self ):
        alpha = [ 1, 2, 3, 4, 5 ]
        sigma = get_norder_paulis(1)
        self.assertRaises( ValueError, pauli_dot_product, alpha, sigma )

    def test_pauli_dot_product_1 ( self ):
        alpha = [ 1, 0, 0, 0 ]
        sigma = get_norder_paulis( 1 )
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), I ) )
        alpha = [ 0, 1, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), X ) )
        alpha = [ 0, 0, 1, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), Y ) )
        alpha = [ 0, 0, 0, 1 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), Z ) )

    def test_pauli_dot_product_2 ( self ):
        alpha = [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        sigma = get_norder_paulis( 2 )
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( I, I ) ) )
        alpha = [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( I, X ) ) )
        alpha = [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( I, Y ) ) )
        alpha = [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( I, Z ) ) )
        alpha = [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( X, I ) ) )
        alpha = [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( X, X ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( X, Y ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( X, Z ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Y, I ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Y, X ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Y, Y ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Y, Z ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Z, I ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Z, X ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Z, Y ) ) )
        alpha = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]
        self.assertTrue( np.allclose( pauli_dot_product( alpha, sigma ), np.kron( Z, Z ) ) )


if __name__ == '__main__':
    tf.test.main()
