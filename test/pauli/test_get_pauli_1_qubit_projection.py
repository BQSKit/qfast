import tensorflow as tf
import numpy      as np

from qfast import *


class TestGetPauli1QubitProjection ( tf.test.TestCase ):

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


if __name__ == '__main__':
    tf.test.main()
