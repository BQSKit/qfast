import tensorflow as tf
import numpy      as np

from qfast import pauli_expansion, I, X, Y, Z, get_norder_paulis
from qfast import hilbert_schmidt_distance
from qfast import get_unitary_from_pauli_coefs, unitary_log_no_i

class TestGetUnitaryFromPauliCoefs ( tf.test.TestCase ):

    def test_get_unitary_from_pauli_coefs_invalid ( self ):
        pauli_coefs = [ 1, 2, 3, 4, 5 ]
        self.assertRaises( ValueError, get_unitary_from_pauli_coefs, pauli_coefs )

    def test_get_unitary_from_pauli_coefs_1 ( self ):
        sigma = get_norder_paulis( 1 )

        for U in sigma:
            pauli_coefs = pauli_expansion( unitary_log_no_i( U ) )
            reU = get_unitary_from_pauli_coefs( pauli_coefs )
            self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
            self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 )
                             and
                             np.allclose( U @ U.conj().T, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 ) )

    # def test_get_unitary_from_pauli_coefs_2 ( self ):
    #     sigma = get_norder_paulis( 2 )

    #     for U in sigma:
    #         pauli_coefs = pauli_expansion( unitary_log_no_i( U ) )
    #         reU = get_unitary_from_pauli_coefs( pauli_coefs )
    #         self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
    #         self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 )
    #                          and
    #                          np.allclose( U @ U.conj().T, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 ) )

    # def test_get_unitary_from_pauli_coefs_3 ( self ):
    #     sigma = get_norder_paulis( 3 )

    #     for U in sigma:
    #         pauli_coefs = pauli_expansion( unitary_log_no_i( U ) )
    #         reU = get_unitary_from_pauli_coefs( pauli_coefs )
    #         self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
    #         self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 )
    #                          and
    #                          np.allclose( U @ U.conj().T, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 ) )

    # def test_get_unitary_from_pauli_coefs_4 ( self ):
    #     sigma = get_norder_paulis( 4 )

    #     for U in sigma:
    #         pauli_coefs = pauli_expansion( unitary_log_no_i( U ) )
    #         reU = get_unitary_from_pauli_coefs( pauli_coefs )
    #         self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
    #         self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 )
    #                          and
    #                          np.allclose( U @ U.conj().T, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 ) )

    # def test_get_unitary_from_pauli_coefs_comb ( self ):
    #     sigma = get_norder_paulis( 4 )

    #     for U1, U2 in zip( sigma, sigma[1:] ):
    #         U = U1 @ U2
    #         pauli_coefs = pauli_expansion( unitary_log_no_i( U ) )
    #         reU = get_unitary_from_pauli_coefs( pauli_coefs )
    #         self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
    #         self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 )
    #                          and
    #                          np.allclose( U @ U.conj().T, np.identity( len( U ) ),
    #                                       rtol = 0, atol = 1e-16 ) )


if __name__ == '__main__':
    tf.test.main()
