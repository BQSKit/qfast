import numpy    as np
import scipy    as sp
import unittest as ut

from qfast.utils import is_unitary, dot_product
from qfast.pauli import get_norder_paulis

class TestIsUnitary ( ut.TestCase ):
    
    def test_is_unitary1 ( self ):
        paulis = get_norder_paulis( 3 )

        for i in range( 10 ):
            alpha = np.random.random( 4 ** 3 )
            U = sp.linalg.expm( 1j * dot_product( alpha, paulis ) )
            self.assertTrue( is_unitary( U, tol = 1e-14 ) )

    def test_is_unitary2 ( self ):
        paulis = get_norder_paulis( 3 )

        for i in range( 10 ):
            alpha = np.random.random( 4 ** 3 )
            U = sp.linalg.expm( 1j * dot_product( alpha, paulis ) )
            U += 1e-13 * np.ones( 8 )
            self.assertTrue( is_unitary( U, tol = 1e-12 ) )
   
    def test_is_unitary_invalid ( self ):
        self.assertFalse( is_unitary( 1j * np.ones( ( 4, 4 ) ) ) )
        self.assertFalse( is_unitary( np.ones( ( 4, 3 ) ) ) )
        self.assertFalse( is_unitary( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_unitary( 1 ) )
        self.assertFalse( is_unitary( "a" ) )


if __name__ == '__main__':
    ut.main()
