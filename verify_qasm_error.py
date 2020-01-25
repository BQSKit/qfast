import argparse

import numpy as np
from qiskit import *


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


def test_unitary ( circ, utry ):
    # Test unitary shape
    num_qubits = len( circ.qubits )
    if utry.shape != (2**num_qubits, 2**num_qubits):
        raise RuntimeError( "Unitary failed verification test: wrong shape." )

    # Test unitary matrix
    backend = BasicAer.get_backend( 'statevector_simulator' )

    for idx in range( 2**num_qubits ):
        # For all standard basis vectors (Have to reverse for qiskit)
        base_vec = np.array( [ int(x == idx) for x in range( 2**num_qubits ) ] )
        revd_idx = int( ('{0:0%db}' % num_qubits).format(idx)[::-1], 2 )
        revd_vec = np.array( [ int(x == revd_idx) for x in range( 2**num_qubits ) ] )

        # Calculate correct probability distribution
        boptions = { 'initial_statevector': revd_vec }
        verf_job = qiskit.execute( circ, backend, backend_options = boptions )
        verf_vec = verf_job.result().get_statevector()

        verf_dist = {}
        for x in range( 2**num_qubits ):
            bit_str = ('{0:0%db}' % num_qubits).format(x)
            bit_str = bit_str[::-1]  # reverse( bit_str )
            bit_prb = np.abs( verf_vec[x] ) ** 2
            verf_dist[ bit_str ] = bit_prb

        # Calculate test probability distribution
        test_vec = utry @ np.reshape( base_vec, ( len( base_vec ), 1 ) )
        test_vec = np.squeeze( test_vec )

        test_dist = {}
        for x in range( 2**num_qubits ):
            bit_str = ('{0:0%db}' % num_qubits).format(x)
            bit_prb = np.abs( test_vec[x] ) ** 2
            test_dist[ bit_str ] = bit_prb

        # Test equality of distributions
        for x in range( 2**num_qubits ):
            bit_str = ('{0:0%db}' % num_qubits).format(x)
            if not np.allclose( verf_dist[ bit_str ], test_dist[ bit_str ] ):
                raise RuntimeError( "Unitary failed verification test: incorrect matrix" )


def hilbert_schmidt_distance ( X, Y ):
    """
    Calculates a distance based on the Hilbert Schmidt inner product.

    Args:
        X: First Operator
        Y: Second Operator

    Returns:
        Error value between X and Y
    """

    if X.shape != Y.shape:
        raise ValueError( "X and Y must have same shape." )

    mat = np.matmul( np.transpose( np.conj( X ) ), Y )
    num = np.abs( np.trace( mat ) ) ** 2
    dem = mat.shape[0] ** 2
    return np.sqrt( 1 - ( num / dem ) )


if __name__ == "__main__":
    description_info = "Convert QASM to a unitary."

    parser = argparse.ArgumentParser( description = description_info )

    parser.add_argument( "qasm_file", type = str,
                         help = "QASM input file" )

    parser.add_argument( "unitary_file", type = str,
                         help = "Unitary output file" )

    args = parser.parse_args()

    circ = QuantumCircuit.from_qasm_file( args.qasm_file )
    circ.remove_final_measurements()
    utry = calc_unitary( flip_circ( circ ) )
    test_unitary( circ, utry )

    target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
    print( hilbert_schmidt_distance( utry, target ) )


