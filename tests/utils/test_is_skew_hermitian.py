import numpy    as np
import unittest as ut

from qfast.utils import is_skew_hermitian, dot_product
from qfast.pauli import get_norder_paulis

class TestIsSkewHermitian ( ut.TestCase ):
    
    def test_is_skew_hermitian ( self ):
        paulis = get_norder_paulis( 3 )

        for i in range( 10 ):
            alpha = 1j * np.random.random( 4 ** 3 )
            self.assertTrue( is_skew_hermitian( dot_product( alpha, paulis ) ) )
    
    def test_is_skew_hermitian_invalid ( self ):
        self.assertFalse( is_skew_hermitian( np.ones( ( 4, 4 ) ) ) )
        self.assertFalse( is_skew_hermitian( np.ones( ( 4, 3 ) ) ) )
        self.assertFalse( is_skew_hermitian( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_skew_hermitian( 1 ) )
        self.assertFalse( is_skew_hermitian( "a" ) )

if __name__ == '__main__':
    ut.main()
