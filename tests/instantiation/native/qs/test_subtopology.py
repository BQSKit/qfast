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


class TestQSSubtopology ( ut.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )
    
    def test_qs_subtopoology_empty ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx q[0], q[1];" in qasm or "cx q[1], q[0]" in qasm )
        self.assertTrue( "cx q[1], q[2];" in qasm or "cx q[2], q[1]" in qasm )
        self.assertFalse( "cx q[0], q[2];" in qasm or "cx q[2], q[0]" in qasm )

    def test_qs_subtopoology_None ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, coupling_graph = None )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx q[0], q[1];" in qasm or "cx q[1], q[0]" in qasm )
        self.assertTrue( "cx q[1], q[2];" in qasm or "cx q[2], q[1]" in qasm )
        self.assertFalse( "cx q[0], q[2];" in qasm or "cx q[2], q[0]" in qasm )

    def test_qs_subtopoology_line ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, coupling_graph = [ ( 0, 1 ), ( 1, 2 ) ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx q[0], q[1];" in qasm or "cx q[1], q[0]" in qasm )
        self.assertTrue( "cx q[1], q[2];" in qasm or "cx q[2], q[1]" in qasm )
        self.assertFalse( "cx q[0], q[2];" in qasm or "cx q[2], q[0]" in qasm )

    def test_qs_subtopoology_angle1 ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, coupling_graph = [ ( 0, 1 ), ( 0, 2 ) ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertTrue( "cx q[0], q[1];" in qasm or "cx q[1], q[0]" in qasm )
        self.assertFalse( "cx q[1], q[2];" in qasm or "cx q[2], q[1]" in qasm )
        self.assertTrue( "cx q[0], q[2];" in qasm or "cx q[2], q[0]" in qasm )

    def test_qs_subtopoology_angle2 ( self ):
        qtool = QSearchTool()
        qasm = qtool.synthesize( self.TOFFOLI, coupling_graph = [ ( 0, 2 ), ( 1, 2 ) ] )
        utry = get_utry( QuantumCircuit.from_qasm_str( qasm ) )
        self.assertTrue( hilbert_schmidt_distance( self.TOFFOLI, utry ) <= 1e-15 )
        self.assertFalse( "cx q[0], q[1];" in qasm or "cx q[1], q[0]" in qasm )
        self.assertTrue( "cx q[1], q[2];" in qasm or "cx q[2], q[1]" in qasm )
        self.assertTrue( "cx q[0], q[2];" in qasm or "cx q[2], q[0]" in qasm )


if __name__ == '__main__':
    ut.main()
