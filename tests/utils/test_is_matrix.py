import numpy    as np
import unittest as ut

from qfast.utils import is_matrix, dot_product
from qfast.pauli import get_norder_paulis

class TestIsMatrix ( ut.TestCase ):
    
    def test_is_matrix1 ( self ):
        paulis = get_norder_paulis( 3 )

        for i in range( 10 ):
            alpha = np.random.random( 4 ** 3 )
            self.assertTrue( is_matrix( dot_product( alpha, paulis ) ) )
   
    def test_is_matrix2 ( self ):
        self.assertTrue( is_matrix( 1j * np.ones( ( 4, 4 ) ) ) )
        self.assertTrue( is_matrix( np.ones( ( 4, 3 ) ) ) )

    def test_is_matrix_invalid ( self ):
        self.assertFalse( is_matrix( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_matrix( 1 ) )
        self.assertFalse( is_matrix( "a" ) )


if __name__ == '__main__':
    ut.main()
