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


def get_utry ( circ ):
    """Converts a qiskit circuit into a numpy unitary."""

    backend = BasicAer.get_backend( 'unitary_simulator' )
    utry = qiskit.execute( circ, backend ).result().get_unitary()
    num_qubits = int( np.log2( len( utry ) ) )
    qubit_order = list( reversed( range( num_qubits ) ) )
    P = calc_permutation_matrix( num_qubits, qubit_order )
    return P @ utry @ P.T



for file in os.listdir():
    if os.path.isfile( file ) and file[ -5 : ] == ".qasm":
        name = file[ : -5 ]
        print( name )
        try:
            circ = qiskit.QuantumCircuit.from_qasm_file( file )
            utry = get_utry( circ )

            with open( name + ".unitary", "w" ) as f:
                np.savetxt( f, utry )
        except:
            print( "wtf" )
