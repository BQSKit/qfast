import numpy    as np
import unittest as ut

from qiskit import *

from qfast import perm
from qfast.instantiation.native.qs import QSearchTool

def get_utry ( circ ):
    """Converts a qiskit circuit into a numpy unitary."""

    backend = qiskit.BasicAer.get_backend( 'unitary_simulator' )
    utry = qiskit.execute( circ, backend ).result().get_unitary()
    num_qubits = int( np.log2( len( utry ) ) )
    qubit_order = tuple( reversed( range( num_qubits ) ) )
    P = perm.calc_permutation_matrix( num_qubits, qubit_order )
    return P @ utry @ P.T


def hilbert_schmidt_distance ( X, Y ):
    """Calculates a Hilbert-Schmidt based distance."""

    if X.shape != Y.shape:
        raise ValueError( "X and Y must have same shape." )

    mat = np.matmul( np.transpose( np.conj( X ) ), Y )
    num = np.abs( np.trace( mat ) )
    dem = mat.shape[0]
    return 1 - ( num / dem )


class TestQSBasisGates ( ut.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_qs_basis_gates_None ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, basis_gates = None )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx" in qasm )
        self.assertTrue( "cz" not in qasm )
        self.assertTrue( "iswap" not in qasm )
        self.assertTrue( "rxx" not in qasm )

    def test_qs_basis_gates_cx ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, basis_gates = [ "cx" ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx" in qasm )
        self.assertTrue( "cz" not in qasm )
        self.assertTrue( "iswap" not in qasm )
        self.assertTrue( "rxx" not in qasm )

    def test_qs_basis_gates_cz ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, basis_gates = [ "cz" ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx" not in qasm )
        self.assertTrue( "cz" in qasm )
        self.assertTrue( "iswap" not in qasm )
        self.assertTrue( "rxx" not in qasm )

    def test_qs_basis_gates_iswap ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, basis_gates = [ "iswap" ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx" in qasm )
        self.assertTrue( "s" in qasm )
        self.assertTrue( "h" in qasm )
        self.assertTrue( "cz" not in qasm )
        self.assertTrue( "rxx" not in qasm )

    def test_qs_basis_gates_rxx ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, basis_gates = [ "rxx" ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx" not in qasm )
        self.assertTrue( "cz" not in qasm )
        self.assertTrue( "iswap" not in qasm )
        self.assertTrue( "rxx" in qasm )


if __name__ == '__main__':
    ut.main()
