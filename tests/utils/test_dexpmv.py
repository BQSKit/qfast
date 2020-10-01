import numpy    as np
import scipy    as sp
import unittest as ut

from qfast.utils import dexpmv, dot_product
from qfast.pauli import get_norder_paulis


def dexpm_exact ( M, dM, term_count = 100 ):
    """
    Exact matrix exponential derivative calculation.

    Rossmann 2002 Theorem 5 Section 1.2
    """

    adjMp = dM
    total = np.zeros( M.shape, dtype = np.complex128 )
    
    for i in range( term_count ):
        total += adjMp
        adjMp  = (M @ adjMp) - (adjMp @ M)
        adjMp *= -1
        adjMp /= i + 2

    F = sp.linalg.expm( M )
    dF = F @ total
    return F, dF


class TestDexpmv ( ut.TestCase ):
    
    def test_dexpmv_single ( self ):
        n = 2
        paulis = get_norder_paulis( n )
        H = dot_product( np.random.random ( 4 ** n ), paulis )

        for p in paulis:
            F0, dF0 = dexpm_exact( H, p )
            F1, dF1 = dexpmv( H, p )

            self.assertTrue( np.allclose( F0, F1 ) )
            self.assertTrue( np.allclose( dF0, dF1 ) )

    def test_dexpmv_vector ( self ):
        n = 2
        paulis = get_norder_paulis( n )
        H = dot_product( np.random.random ( 4 ** n ), paulis )

        dFs0 = []
        for p in paulis:
            _, dF = dexpm_exact( H, p )
            dFs0.append( dF )

        dFs0 = np.array( dFs0 )

        _, dFs1 = dexpmv( H, paulis )

        self.assertTrue( np.allclose( dFs0, dFs1 ) )

    def test_dexpmv_invalid ( self ):
        self.assertRaises( Exception, dexpmv, 0, 0 )
        self.assertRaises( Exception, dexpmv, 0, [1, 0] )
        self.assertRaises( Exception, dexpmv, [1, 0], 0 )
        self.assertRaises( Exception, dexpmv, [1, 0], [1, 0] )

        I = np.identity( 2 )
        self.assertRaises( Exception, dexpmv, I, 0 )
        self.assertRaises( Exception, dexpmv, 0, I )
        self.assertRaises( Exception, dexpmv, I, [1, 0] )
        self.assertRaises( Exception, dexpmv, [1, 0], I )


if __name__ == '__main__':
    ut.main()
