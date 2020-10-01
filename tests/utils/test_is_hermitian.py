import numpy    as np
import unittest as ut

from qfast.utils import is_hermitian, dot_product
from qfast.pauli import get_norder_paulis

class TestIsHermitian ( ut.TestCase ):
    
    def test_is_hermitian ( self ):
        paulis = get_norder_paulis( 3 )

        for i in range( 10 ):
            alpha = np.random.random( 4 ** 3 )
            self.assertTrue( is_hermitian( dot_product( alpha, paulis ) ) )
    
    def test_is_hermitian_invalid ( self ):
        self.assertFalse( is_hermitian( 1j * np.ones( ( 4, 4 ) ) ) )
        self.assertFalse( is_hermitian( np.ones( ( 4, 3 ) ) ) )
        self.assertFalse( is_hermitian( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_hermitian( 1 ) )
        self.assertFalse( is_hermitian( "a" ) )

if __name__ == '__main__':
    ut.main()
