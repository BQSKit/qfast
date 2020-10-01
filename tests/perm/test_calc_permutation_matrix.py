import numpy    as np
import scipy    as sp
import unittest as ut

from qfast.perm import calc_permutation_matrix


class TestCalcPermutationMatrix ( ut.TestCase ):
    
    def test_calc_permutation_matrix ( self ):
        swap_012 = np.array( [ [ 1, 0, 0, 0 ],
                               [ 0, 0, 1, 0 ],
                               [ 0, 1, 0, 0 ],
                               [ 0, 0, 0, 1 ] ] )

        perm = calc_permutation_matrix( 2, (1, 0) )
        self.assertTrue( np.allclose( perm, swap_012 ) )

        perm = calc_permutation_matrix( 2, (1,) )
        self.assertTrue( np.allclose( perm, swap_012 ) )

        perm = calc_permutation_matrix( 2, (0, 1) )
        self.assertTrue( np.allclose( perm, np.identity(4) ) )

    def test_calc_permutation_matrix_big ( self ):
        I = np.identity( 2, dtype = np.complex128 )
        II = np.kron( I, I )
        IIII = np.kron( II, II )
        X = np.array( [ [ 0, 1 ], [ 1, 0 ] ], dtype = np.complex128 )
        XX = np.kron( X, X )
        XXI = np.kron( XX, I )
        IXX = np.kron( I, XX )
        IIXX = np.kron( I, IXX )
        IX = np.kron( I, X )
        IXIX = np.kron( IX, IX )
        XXXX = np.kron( XX, XX )
        IXIXIXIX = np.kron( IXIX, IXIX )

        U0 = sp.linalg.expm( -1j * IXX )
        U1 = sp.linalg.expm( -1j * XXI )
        P = calc_permutation_matrix( 3, ( 1, 2 ) )
        self.assertTrue( np.allclose( U0, P @ U1 @ P.T ) )

        U0 = sp.linalg.expm( -1j * IIXX )
        U1 = sp.linalg.expm( -1j * IXIX )
        P = calc_permutation_matrix( 4, ( 0, 2 ) )
        self.assertTrue( np.allclose( U0, P @ U1 @ P.T ) )

        U0 = sp.linalg.expm( -1j * IXIXIXIX )
        U1 = sp.linalg.expm( -1j * np.kron( XXXX, IIII ) )
        P = calc_permutation_matrix( 8, ( 1, 3, 5, 7 ) )
        self.assertTrue( np.allclose( U0, P @ U1 @ P.T ) )

    def test_calc_permutation_matrix_invalid ( self ):
        self.assertRaises( TypeError, calc_permutation_matrix, 4, "a" )
        self.assertRaises( TypeError, calc_permutation_matrix, 4, ( "a" ) )
        self.assertRaises( TypeError, calc_permutation_matrix, 4, [ 0, 1 ] )
        self.assertRaises( TypeError, calc_permutation_matrix, "a", ( 0, 1 ) )


if __name__ == '__main__':
    ut.main()

