"""
qasm2unitary.py

Converts a QASM file to a unitary file using QISKit.
QISKit has an unusual method of flipping the tensor order of qubits.
As such, we flip the qubits in a circuit before feeding it to QISKit
to produce a unitary file. This creates some headaches.
"""

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
        elif gate[0].name == 'rx':
            circ_flip.rx( *gate[0].params, num_qubits - 1 - gate[1][0].index )
        elif gate[0].name == 'ry':
            circ_flip.ry( *gate[0].params, num_qubits - 1 - gate[1][0].index )
        elif gate[0].name == 'rz':
            circ_flip.rz( *gate[0].params, num_qubits - 1 - gate[1][0].index )
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


if __name__ == "__main__":
    description_info = "Convert QASM to a unitary."

    parser = argparse.ArgumentParser( description = description_info )

    parser.add_argument( "qasm_file", type = str,
                         help = "QASM input file" )

    parser.add_argument( "unitary_file", type = str,
                         help = "Unitary output file" )

    parser.add_argument( "-t", "--test", action = 'store_true',
                         help = "Test unitary" )

    args = parser.parse_args()

    circ = QuantumCircuit.from_qasm_file( args.qasm_file )
    circ.remove_final_measurements()
    utry = calc_unitary( flip_circ( circ ) )

    if args.test:
        test_unitary( circ, utry )

    np.savetxt( args.unitary_file, utry )
