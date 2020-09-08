import numpy    as np
import unittest as ut

from qfast.pauli import get_pauli_n_qubit_projection


class TestGetPauliNQubitProjection ( ut.TestCase ):

    def in_array( self, needle, haystack ):
        for elem in haystack:
            if np.allclose( elem, needle ):
                return True

        return False

    def test_get_pauli_1_qubit_proj_n1 ( self ):
        num_qubits = -1
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [2] )

    def test_get_pauli_1_qubit_proj_o1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [6] )

    def test_get_pauli_1_qubit_proj_q1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [-1] )

    def test_get_pauli_1_qubit_proj_3_0 ( self ):
        num_qubits = 3
        qubit_proj = 0
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( X, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), I ), paulis ) )

    def test_get_pauli_1_qubit_proj_3_1 ( self ):
        num_qubits = 3
        qubit_proj = 1
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( I, X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), I ), paulis ) )

    def test_get_pauli_1_qubit_proj_3_2 ( self ):
        num_qubits = 3
        qubit_proj = 2
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), I ), paulis ) )

    def test_get_pauli_1_qubit_proj_4_0 ( self ):
        num_qubits = 4
        qubit_proj = 0
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( X, I ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Y, I ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Z, I ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), I ), paulis ) )

    def test_get_pauli_1_qubit_proj_4_1 ( self ):
        num_qubits = 4
        qubit_proj = 1
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, X ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, Y ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, Z ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), I ), paulis ) )

    def test_get_pauli_1_qubit_proj_4_2 ( self ):
        num_qubits = 4
        qubit_proj = 2
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), I ), paulis ) )

    def test_get_pauli_1_qubit_proj_4_3 ( self ):
        num_qubits = 4
        qubit_proj = 3
        paulis = get_pauli_n_qubit_projection( num_qubits, [ qubit_proj ] )
        self.assertTrue( len( paulis ) == 4 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), I ), paulis ) )

    def test_get_pauli_2_qubit_proj_n1 ( self ):
        num_qubits = -1
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [2, 3] )

    def test_get_pauli_2_qubit_proj_o1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [6, 1] )

    def test_get_pauli_2_qubit_proj_q1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [-1, 1] )

    def test_get_pauli_2_qubit_proj_2o1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [1, 6] )

    def test_get_pauli_2_qubit_proj_2q1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [1, -1] )

    def test_get_pauli_2_qubit_proj_q0eq1 ( self ):
        num_qubits = 5
        self.assertRaises( ValueError, get_pauli_n_qubit_projection, num_qubits, [1, 1] )

    def test_get_pauli_2_qubit_proj_3_0_1 ( self ):
        num_qubits = 3
        qubit_pro1 = 0
        qubit_pro2 = 1
        paulis = get_pauli_n_qubit_projection( num_qubits, [qubit_pro1, qubit_pro2] )
        self.assertTrue( len( paulis ) == 16 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( X, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( X, X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, X ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( X, Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Y ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( X, Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Z ), I ), paulis ) )

    def test_get_pauli_2_qubit_proj_3_0_2 ( self ):
        num_qubits = 3
        qubit_pro1 = 0
        qubit_pro2 = 2
        paulis = get_pauli_n_qubit_projection( num_qubits, [qubit_pro1, qubit_pro2] )
        self.assertTrue( len( paulis ) == 16 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( X, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( X, I ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, I ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, I ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), X ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( X, I ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, I ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, I ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), Y ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( X, I ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Y, I ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( Z, I ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), Z ), paulis ) )

    def test_get_pauli_2_qubit_proj_3_1_2 ( self ):
        num_qubits = 3
        qubit_pro1 = 1
        qubit_pro2 = 2
        paulis = get_pauli_n_qubit_projection( num_qubits, [qubit_pro1, qubit_pro2] )
        self.assertTrue( len( paulis ) == 16 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( I, X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( I, X ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Y ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Z ), X ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), X ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( I, X ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Y ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Z ), Y ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), Y ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( I, X ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Y ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, Z ), Z ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( I, I ), Z ), paulis ) )

    def test_get_pauli_2_qubit_proj_4_0_2 ( self ):
        num_qubits = 4
        qubit_pro1 = 0
        qubit_pro2 = 2
        paulis = get_pauli_n_qubit_projection( num_qubits, [qubit_pro1, qubit_pro2] )
        self.assertTrue( len( paulis ) == 16 )

        X = np.array( [[0, 1], [1, 0]], dtype = np.complex128 )
        Y = np.array( [[0, -1j], [1j, 0]], dtype = np.complex128 )
        Z = np.array( [[1, 0], [0, -1]], dtype = np.complex128 )
        I = np.array( [[1, 0], [0, 1]], dtype = np.complex128 )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( X, I ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Y, I ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Z, I ), I ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), I ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( X, I ), X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Y, I ), X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Z, I ), X ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), X ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( X, I ), Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Y, I ), Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Z, I ), Y ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), Y ), I ), paulis ) )

        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( X, I ), Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Y, I ), Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( Z, I ), Z ), I ), paulis ) )
        self.assertTrue( self.in_array( np.kron( np.kron( np.kron( I, I ), Z ), I ), paulis ) )


if __name__ == '__main__':
    ut.main()
