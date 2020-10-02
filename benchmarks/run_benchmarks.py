"""
This scripts tests qfast with all .unitary files in
the current directory.
"""

import os
import logging
import pickle
import signal
import subprocess

from datetime import date
from io import StringIO
from timeit import default_timer as timer

import numpy as np

import qfast
from qfast import synthesize
from qfast.perm import calc_permutation_matrix

from qiskit import *

logger = logging.getLogger( "qfast" )


def get_exp_header():
    """Creates the experiment log header."""

    v = qfast.__version__
    g = subprocess.check_output( ['git', 'rev-parse',
                                  '--short', 'HEAD'] )
    g = g.decode('ascii').strip()
    return str(v) + " - " + str(g)


def get_utry ( circ ):
    """Converts a qiskit circuit into a numpy unitary."""

    backend = BasicAer.get_backend( 'unitary_simulator' )
    utry = qiskit.execute( circ, backend ).result().get_unitary()
    num_qubits = int( np.log2( len( utry ) ) )
    qubit_order = tuple( reversed( range( num_qubits ) ) )
    P = calc_permutation_matrix( num_qubits, qubit_order )
    return P @ utry @ P.T


def hilbert_schmidt_distance ( X, Y ):
    """Calculates a Hilbert-Schmidt based distance."""

    if X.shape != Y.shape:
        raise ValueError( "X and Y must have same shape." )

    mat = np.matmul( np.transpose( np.conj( X ) ), Y )
    num = np.abs( np.trace( mat ) )
    dem = mat.shape[0]
    return 1 - ( num / dem )


def get_error ( utry_in, qasm ):
    """Calculates the experiment error."""

    circ = QuantumCircuit.from_qasm_str( qasm )
    circ.remove_final_measurements()
    utry_out = get_utry( circ )
    return hilbert_schmidt_distance( utry_in, utry_out )


def termTrial ( signal_number, frame ):
    """Terminate a Trial"""

    msg = "Error"

    if signal_number == signal.SIGINT:
        msg = "Manually Interrupted"

    if signal_number == signal.SIGALRM:
        msg = "Timed-out"

    logger.error( msg )
    raise Exception( msg )


# Register Signal Handlers
signal.signal( signal.SIGALRM, termTrial )
signal.signal( signal.SIGINT, termTrial )


# Initialize Data Folders
if not os.path.isdir( ".checkpoints" ):
    os.makedirs( ".checkpoints" )

start_date = str( date.today() )
exp_folder = ".checkpoints/" + start_date + "/"

if not os.path.isdir( exp_folder ):
    os.makedirs( exp_folder )

data = {}

# Set Random Seed
np.random.seed(21211411)

hierarchy_fn = lambda x : 3 if x >= 7 else 2

for file in os.listdir():
    if os.path.isfile( file ) and file[ -8 : ] == ".unitary":
        name = file[ : -8 ]
        utry = np.loadtxt( file, dtype = np.complex128 )
        num_qubits = int( np.log2( len( utry ) ) )
        stream = StringIO()
        handler = logging.StreamHandler( stream )
        handler.setLevel( logging.DEBUG )
        logger.setLevel( logging.DEBUG )
        logger.addHandler( handler )

        logger.info( "-" * 40 )
        logger.info( get_exp_header() )
        logger.info( start_date )
        logger.info( name )
        logger.info( "-" * 40 )

        timeout = False
        timeouts = { 3: 10*60, 4: 20*60, 5: 45*60, 6: 90*60 }
        signal.alarm( timeouts[ num_qubits ] )

        # Run Benchmark
        start = timer()

        try:
            qasm = synthesize( utry, model = "PermModel", hierarchy_fn = hierarchy_fn )
        except Exception as ex:
            print( ex )
            timeout = True
            pass

        end = timer()

        handler.flush()
        logger.removeHandler( handler )
        handler.close()
        log_out = stream.getvalue()

        if timeout == False:
            data[ name ] = ( num_qubits, qasm.count( "cx" ), end - start,
                             get_error( utry, qasm ), qasm, log_out )
        else: 
            data[ name ] = ( num_qubits,  log_out )
            

        # Save data
        filenum = 0
        filename = exp_folder + name + str( filenum ) + ".dat"
        while os.path.isfile( filename ):
            filenum += 1
            filename = exp_folder + name + str( filenum ) + ".dat"
        file = open( filename, 'wb' )
        pickle.dump( data[ name ], file )

for test in data:
    print( "-" * 40 )
    print( test )
    print( "CX count:", data[test][1] )
    print( "Time (s):", data[test][2] )
    print( "Error:", data[test][3] )
print( "-" * 40 )
