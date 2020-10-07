"""Example synthesis of a 4-qubit QFT program using QFAST."""


import numpy as np

from qfast import synthesize


# You can enable verbose logging with the following two lines.
# import logging
# logging.getLogger( "qfast" ).setLevel( logging.DEBUG )

# Read the qft4 file in
qft4 = np.loadtxt( "qft4.unitary", dtype = np.complex128 )

# Synthesize the qft4 unitary and print the resulting qasm code
print( synthesize( qft4 ) )
