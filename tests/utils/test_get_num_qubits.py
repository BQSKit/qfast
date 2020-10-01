import numpy    as np
import unittest as ut

from qfast.utils import get_num_qubits


class TestGetNumQubits ( ut.TestCase ):
    
    def test_num_qubits ( self ):
        for i in range( 4 ):
            M = np.identity( 2 ** i )
            self.assertTrue( get_num_qubits( M ) == i )
    
    def test_num_qubits_invalid ( self ):
        M0 = np.ones( ( 3, 2 ) )
        M1 = "a"
        M2 = 3

        self.assertRaises( TypeError, get_num_qubits, M0 )
        self.assertRaises( TypeError, get_num_qubits, M1 )
        self.assertRaises( TypeError, get_num_qubits, M2 )


if __name__ == '__main__':
    ut.main()
