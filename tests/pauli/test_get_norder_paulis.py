import numpy    as np
import unittest as ut

from qfast.pauli import get_norder_paulis


class TestGetNorderPaulis ( ut.TestCase ):

    def in_array( self, needle, haystack ):
        for elem in haystack:
            if np.allclose( elem, needle ):
                return True

        return False
    
    def test_get_norder_paulis_n1 ( self ):
        num_qubits = -1
        self.assertRaises( ValueError, get_norder_paulis, num_qubits )

    def test_get_norder_paulis_0 ( self ):
        num_qubits = 0
        paulis = get_norder_paulis( num_qubits )
        self.assertTrue( len( paulis ) == 4 ** num_qubits )

        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( I, paulis ) )

    def test_get_norder_paulis_1 ( self ):
        num_qubits = 1
        paulis = get_norder_paulis( num_qubits )
        self.assertTrue( len( paulis ) == 4 ** num_qubits )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( X, paulis ) )
        self.assertTrue( self.in_array( Y, paulis ) )
        self.assertTrue( self.in_array( Z, paulis ) )
        self.assertTrue( self.in_array( I, paulis ) )

    def test_get_norder_paulis_2 ( self ):
        num_qubits = 2
        paulis = get_norder_paulis( num_qubits )
        self.assertTrue( len( paulis ) == 4 ** num_qubits )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( X, X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, I ), paulis ) )

    def test_get_norder_paulis_3 ( self ):
        num_qubits = 3
        paulis = get_norder_paulis( num_qubits )
        self.assertTrue( len( paulis ) == 4 ** num_qubits )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( X, np.kron( X, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( X, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( X, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( X, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Y, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Y, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Y, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Y, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Z, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Z, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Z, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( Z, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( I, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( I, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( I, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( X, np.kron( I, I ) ), paulis ) )

        self.assertTrue( self.in_array( np.kron( Y, np.kron( X, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( X, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( X, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( X, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Y, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Y, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Y, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Y, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Z, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Z, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Z, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( Z, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( I, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( I, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( I, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Y, np.kron( I, I ) ), paulis ) )

        self.assertTrue( self.in_array( np.kron( Z, np.kron( X, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( X, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( X, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( X, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Y, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Y, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Y, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Y, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Z, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Z, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Z, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( Z, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( I, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( I, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( I, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( Z, np.kron( I, I ) ), paulis ) )

        self.assertTrue( self.in_array( np.kron( I, np.kron( X, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( X, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( X, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( X, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Y, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Y, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Y, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Y, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Z, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Z, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Z, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( Z, I ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( I, X ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( I, Y ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( I, Z ) ), paulis ) )
        self.assertTrue( self.in_array( np.kron( I, np.kron( I, I ) ), paulis ) )


if __name__ == '__main__':
    ut.main()
