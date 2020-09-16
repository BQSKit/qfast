import os
import logging
import numpy as np

from qfast import synthesize
from qiskit import *

from qfast.perm import calc_permutation_matrix

from timeit import default_timer as timer

from io import StringIO
import pickle

def get_utry ( circ ):
    backend = BasicAer.get_backend( 'unitary_simulator' )
    utry = qiskit.execute( circ, backend ).result().get_unitary()
    num_qubits = int( np.log2( len( utry ) ) )
    P = calc_permutation_matrix( num_qubits, list( reversed( range( num_qubits ) ) ) )
    return P @ utry @ P.T


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
    num = np.abs( np.trace( mat ) )
    dem = mat.shape[0]
    return 1 - ( num / dem )


def get_error ( utry_in, qasm ):
    circ = QuantumCircuit.from_qasm_str( qasm )
    circ.remove_final_measurements()
    utry_out = get_utry( circ )
    return hilbert_schmidt_distance( utry_in, utry_out )


data = {}

if not os.path.isdir( ".checkpoints" ):
    os.makedirs( ".checkpoints" )

experiment_info = "First experiment testing experiment.py"
qfast_version = "2.0.dev - 09933fe"
np.random.seed(0)
for file in os.listdir():
    if os.path.isfile( file ) and file[ -8 : ] == ".unitary":
        name = file[ : -8 ]
        utry = np.loadtxt( file, dtype = np.complex128 )
        stream = StringIO()
        handler = logging.StreamHandler( stream )
        logger = logging.getLogger( "qfast" )
        handler.setLevel( logging.DEBUG )
        logger.setLevel( logging.DEBUG )
        logger.addHandler( handler )

        logger.info( "-" * 40 )
        logger.info( f"-- Starting synthesis: {name:<15} --" )
        logger.info( f"-- Version: {qfast_version:<25} --" )
        logger.info( "-" * 14 + " Experiment " + "-" * 14 )
        logger.info( experiment_info )
        logger.info( "-" * 40 )

        start = timer()
        hierarchy_fn = lambda x : 3 if x >= 6 else 2
        qasm = synthesize( utry, model = "TestModel", hierarchy_fn = hierarchy_fn )
        end = timer()

        handler.flush()
        logger.removeHandler( handler )
        handler.close()
        log_out = stream.getvalue()

        num_qubits = int( np.log2( len( utry ) ) )
        data[ name ] = ( num_qubits, qasm.count( "cx" ), end - start,
                         get_error( utry, qasm ), qasm, log_out )
       
        file = open( f".checkpoints/{name}.dat", 'wb' )
        pickle.dump( data[ name ], file )

for test in data:
    print( "-" * 40 )
    print( test )
    print( "CX count:", test[data][1] )
    print( "Time (s):", test[data][2] )
    print( "Error:", test[data][3] )
print( "-" * 40 )
