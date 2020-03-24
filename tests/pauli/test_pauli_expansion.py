import tensorflow as tf
import numpy      as np

from qfast import pauli_expansion, I, X, Y, Z, get_norder_paulis
from qfast import hilbert_schmidt_distance


class TestPauliExpansion ( tf.test.TestCase ):

    def test_pauli_expansion_invalid ( self ):
        H = np.array( [ [ 0, 1 ], [ 1, 1e-15j ] ] )
        self.assertRaises( ValueError, pauli_expansion, H )

    def test_pauli_expansion_valid_1 ( self ):
        sigma = get_norder_paulis( 1 )

        for H in sigma:
            alpha = pauli_expansion( H )
            reH = np.sum( [ a*p for a, p in zip( alpha, sigma ) ], 0 )
            self.assertTrue( hilbert_schmidt_distance( H, reH ) <= 1e-16 )

    def test_pauli_expansion_valid_2 ( self ):
        sigma = get_norder_paulis( 2 )

        for H in sigma:
            alpha = pauli_expansion( H )
            reH = np.sum( [ a*p for a, p in zip( alpha, sigma ) ], 0 )
            self.assertTrue( hilbert_schmidt_distance( H, reH ) <= 1e-16 )

    def test_pauli_expansion_valid_3 ( self ):
        sigma = get_norder_paulis( 3 )

        for H in sigma:
            alpha = pauli_expansion( H )
            reH = np.sum( [ a*p for a, p in zip( alpha, sigma ) ], 0 )
            self.assertTrue( hilbert_schmidt_distance( H, reH ) <= 1e-16 )

    def test_pauli_expansion_valid_4 ( self ):
        sigma = get_norder_paulis( 4 )

        for H in sigma:
            alpha = pauli_expansion( H )
            reH = np.sum( [ a*p for a, p in zip( alpha, sigma ) ], 0 )
            self.assertTrue( hilbert_schmidt_distance( H, reH ) <= 1e-16 )

    def test_pauli_expansion_valid_comb ( self ):
        sigma = get_norder_paulis( 4 )
        sqrt2 = np.sqrt(2) / 2

        for H1, H2 in zip( sigma, sigma[1:] ):
            H = sqrt2 * H1 + sqrt2 * H2
            alpha = pauli_expansion( H )
            reH = np.sum( [ a*p for a, p in zip( alpha, sigma ) ], 0 )
            self.assertTrue( hilbert_schmidt_distance( H, reH ) <= 1e-16 )


if __name__ == '__main__':
    tf.test.main()
