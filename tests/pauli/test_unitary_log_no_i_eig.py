import tensorflow as tf
import numpy      as np
import scipy.linalg as la

from qfast import I, X, Y, Z, get_norder_paulis, unitary_log_no_i_eig
from qfast import hilbert_schmidt_distance


class TestUnitaryLogNoIEig ( tf.test.TestCase ):

    def test_unitary_log_no_i_eig_invalid ( self ):
        U = np.array( [ [ 0, 1 ], [ 1, 1e-13j ] ] )
        self.assertRaises( ValueError, unitary_log_no_i_eig, U )

    def test_unitary_log_no_i_eig_valid_1 ( self ):
        sigma = get_norder_paulis( 1 )

        for U in sigma:
            H = unitary_log_no_i_eig( U )
            self.assertTrue( np.allclose( H, H.conj().T, rtol = 0, atol = 1e-15 ) )
            reU = la.expm( 1j * H )
            self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
            self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 )
                             and
                             np.allclose( U @ U.conj().T, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 ) )

    def test_unitary_log_no_i_eig_valid_2 ( self ):
        sigma = get_norder_paulis( 2 )

        for U in sigma:
            H = unitary_log_no_i_eig( U )
            self.assertTrue( np.allclose( H, H.conj().T, rtol = 0, atol = 1e-15 ) )
            reU = la.expm( 1j * H )
            self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
            self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 )
                             and
                             np.allclose( U @ U.conj().T, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 ) )

    def test_unitary_log_no_i_eig_valid_3 ( self ):
        sigma = get_norder_paulis( 3 )

        for U in sigma:
            H = unitary_log_no_i_eig( U )
            self.assertTrue( np.allclose( H, H.conj().T, rtol = 0, atol = 1e-15 ) )
            reU = la.expm( 1j * H )
            self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
            self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 )
                             and
                             np.allclose( U @ U.conj().T, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 ) )

    def test_unitary_log_no_i_eig_valid_4 ( self ):
        sigma = get_norder_paulis( 4 )

        for U in sigma:
            H = unitary_log_no_i_eig( U )
            self.assertTrue( np.allclose( H, H.conj().T, rtol = 0, atol = 1e-15 ) )
            reU = la.expm( 1j * H )
            self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
            self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 )
                             and
                             np.allclose( U @ U.conj().T, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 ) )

    def test_unitary_log_no_i_eig_valid_comp ( self ):
        sigma = get_norder_paulis( 4 )

        for U1, U2 in zip( sigma, sigma[1:] ):
            U = U1 @ U2
            H = unitary_log_no_i_eig( U )
            self.assertTrue( np.allclose( H, H.conj().T, rtol = 0, atol = 1e-15 ) )
            reU = la.expm( 1j * H )
            self.assertTrue( hilbert_schmidt_distance( U, reU ) <= 1e-16 )
            self.assertTrue( np.allclose( U.conj().T @ U, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 )
                             and
                             np.allclose( U @ U.conj().T, np.identity( len( U ) ),
                                          rtol = 0, atol = 1e-16 ) )


if __name__ == '__main__':
    tf.test.main()
