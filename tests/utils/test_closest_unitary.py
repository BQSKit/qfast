import numpy    as np
import unittest as ut

from qfast.utils import closest_unitary, is_unitary


class TestClosestUnitary ( ut.TestCase ):
    
    def test_closest_unitary_invalid ( self ):
        invalid_matrix1 = 1
        invalid_matrix2 = np.ones( ( 4, ) )
        invalid_matrix3 = np.ones( ( 2, 3 ) )

        self.assertRaises( TypeError, closest_unitary, invalid_matrix1 )
        self.assertRaises( TypeError, closest_unitary, invalid_matrix2 )
        self.assertRaises( TypeError, closest_unitary, invalid_matrix3 )

    def test_closest_unitary_valid_unitary1 ( self ):
        valid_matrix = np.array( [ [ 0, 1 ], [ 1, 0 ] ] )

        out_matrix = closest_unitary( valid_matrix )

        self.assertTrue( valid_matrix.shape == out_matrix.shape )
        self.assertTrue( np.linalg.norm( out_matrix - valid_matrix ) <= 1e-16 )
        self.assertTrue( is_unitary( out_matrix ) )

    def test_closest_unitary_valid_unitary2 ( self ):
        valid_matrix = np.identity( 4 )

        out_matrix = closest_unitary( valid_matrix )

        self.assertTrue( valid_matrix.shape == out_matrix.shape )
        self.assertTrue( np.linalg.norm( out_matrix - valid_matrix ) <= 1e-16 )
        self.assertTrue( is_unitary( out_matrix ) )

    def test_closest_unitary_valid_nonunitary1 ( self ):
        valid_matrix = np.identity( 4 ) + ( 1e-3 * np.ones( ( 4, 4 ) ) )

        out_matrix = closest_unitary( valid_matrix )

        self.assertTrue( valid_matrix.shape == out_matrix.shape )
        self.assertTrue( is_unitary( out_matrix ) )
        self.assertTrue( np.linalg.norm( out_matrix - valid_matrix ) <= 4 * 1e-3 )


if __name__ == '__main__':
    ut.main()
