"""
QFAST Benchmarking Script

This script runs qfast on all ".unitary" files in the current directory.
"""

import os
import pickle
import signal
import logging
import traceback
import subprocess
from io import StringIO
from timeit import default_timer as timer
from datetime import date

import numpy as np
import qiskit

import qfast
from qfast import synthesize
from qfast import perm
from qfast import utils


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


def get_error ( utry_in, qasm ):
    """Calculates the experiment error."""

    circ = qiskit.QuantumCircuit.from_qasm_str( qasm )
    circ.remove_final_measurements()
    utry_out = get_utry( circ )
    return hilbert_schmidt_distance( utry_in, utry_out )


class SolutionTree():
    """Class that tracks intermediate and partial solutions."""

    def __init__ ( self, utry ):
        self.utry = utry
        self.intermediates = []
        self.partials = [ [] ]

    def add_intermediate ( self, intermediate ):
        """Stores an intermediate solution in the tree."""
        self.intermediates.append( intermediate )
        self.partials.append( [] )

    def add_partial ( self, partial ):
        """Stores a partial solution in the tree."""
        self.partials[ len( self.intermediates ) ].append( partial )


class TrialTerminatedException ( Exception ):
    """Custom timeout or interrupt Exception."""

def term_trial ( signal_number, frame ):
    """Terminate a Trial"""

    msg = "Error"

    if signal_number == signal.SIGINT:
        msg = "Manually Interrupted"

    if signal_number == signal.SIGALRM:
        msg = "Timed-out"

    logger.error( msg )
    raise TrialTerminatedException()


def run_tests():
    # Register Signal Handlers
    signal.signal( signal.SIGALRM, term_trial )
    signal.signal( signal.SIGINT, term_trial )

    # Initialize Data Folders
    if not os.path.isdir( ".checkpoints" ):
        os.makedirs( ".checkpoints" )

    start_date = str( date.today() )
    exp_folder = ".checkpoints/" + start_date + "/"

    if not os.path.isdir( exp_folder ):
        os.makedirs( exp_folder )

    data = {}

    hierarchy_fn = lambda x : 3 if x > 6 else 2

    for file in os.listdir():
        if os.path.isfile( file ) and file[ -8 : ] == ".unitary":
            name = file[ : -8 ]
            utry = np.loadtxt( file, dtype = np.complex128 )
            num_qubits = utils.get_num_qubits( utry )

            # Record QFAST's logger
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

            soltree = SolutionTree( utry )

            timeout = False
            timeouts = { 3: 10*60, 4: 20*60, 5: 45*60, 6: 90*60, 7: 360*60 }
            signal.alarm( timeouts[ num_qubits ] )

            # Set Random Seed
            np.random.seed(21211411)

            # Run Benchmark
            start = timer()

            try:
                qasm = synthesize( utry, model = "SoftPauliModel",
                                   hierarchy_fn = hierarchy_fn,
                                   intermediate_solution_callback = soltree.add_intermediate,
                                   model_options = {
                                       "partial_solution_callback": soltree.add_partial,
                                       "success_threshold": 1e-4
                                   } )

            except TrialTerminatedException:
                timeout = True
            except:
                logger.error( "Benchmark %s encountered error during execution."
                              % name )
                logger.error( traceback.format_exc() )
                traceback.print_exc()

            end = timer()

            handler.flush()
            logger.removeHandler( handler )
            handler.close()
            log_out = stream.getvalue()

            if not timeout:
                data[ name ] = ( num_qubits, qasm.count( "cx" ),
                                 end - start, get_error( utry, qasm ),
                                 qasm, log_out, soltree )
            else:
                data[ name ] = ( num_qubits,  log_out, soltree )

            # Save data
            filenum = 0
            filename = exp_folder + name + str( filenum ) + ".dat"
            while os.path.isfile( filename ):
                filenum += 1
                filename = exp_folder + name + str( filenum ) + ".dat"
            file = open( filename, 'wb' )
            pickle.dump( data[ name ], file )

    # Print Summary
    for test in data:
        if len( data[test] ) < 6:
            continue
        print( "-" * 40 )
        print( test )
        print( "CX count:", data[test][1] )
        print( "Time (s):", data[test][2] )
        print( "Error:", data[test][3] )
    print( "-" * 40 )


if __name__ == "__main__":
    run_tests()

