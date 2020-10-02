import numpy    as np
import unittest as ut

from qiskit import *

from qfast.instantiation.native.kak import KAKTool


def flip_circ ( circ ):
    num_qubits = len( circ.qubits )

    if num_qubits == 1:
        raise ValueError( "Cannot flip 1-qubit circuit." )

    circ_flip = QuantumCircuit( num_qubits )
    for gate in circ.data:
        if gate[0].name == 'cx':
            circ_flip.cx( num_qubits - 1 - gate[1][0].index, num_qubits - 1 - gate[1][1].index )
        elif gate[0].name == 'u3':
            circ_flip.u3( *gate[0].params, num_qubits - 1 - gate[1][0].index )
        elif gate[0].name == 'u2':
            circ_flip.u2( *gate[0].params, num_qubits - 1 - gate[1][0].index )
        elif gate[0].name == 'u1':
            circ_flip.u1( *gate[0].params, num_qubits - 1 - gate[1][0].index )
        elif gate[0].name == 'barrier':
            continue
        else:
            raise ValueError( "Cannot flip circuit not in {U1, U2, U3, CX} basis." )
    return circ_flip


def calc_unitary ( circ ):
    backend = BasicAer.get_backend( 'unitary_simulator' )
    return qiskit.execute( circ, backend ).result().get_unitary()


class TestKakSynthesize ( ut.TestCase ):

    CNOT = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    INVALID = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_kak_synthesize_invalid ( self ):
        tool = KAKTool()
        self.assertRaises( TypeError, tool.synthesize, 1 )
        self.assertRaises( TypeError, tool.synthesize, np.array( [ 0, 1 ] ) )
        self.assertRaises( TypeError, tool.synthesize, np.array( [ [ [ 0 ] ] ] ) )

        self.assertRaises( TypeError, tool.synthesize, self.INVALID )

        invalid_utry_matrix = np.copy( self.CNOT )
        invalid_utry_matrix[2][2] = 137.+0.j

        self.assertRaises( TypeError, tool.synthesize, invalid_utry_matrix )

        large_utry = np.kron( np.identity( 2 ), self.CNOT ) 
        self.assertRaises( ValueError, tool.synthesize, large_utry )

    def test_kak_synthesize_valid ( self ):
        tool = KAKTool()
        qasm = tool.synthesize( self.CNOT )
        utry = calc_unitary( flip_circ( QuantumCircuit.from_qasm_str( qasm ) ) )
        self.assertTrue( np.linalg.norm( self.CNOT - utry ) <= 1e-15 )


if __name__ == '__main__':
    ut.main()

