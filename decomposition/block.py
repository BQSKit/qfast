"""
This module implements the Block Class.

A block is a unitary operation applied to a set of qubits.
"""

import numpy as np

from timeit import default_timer as timer

from .pauli import get_unitary_from_pauli_coefs
from .pauli import unitary_log_no_i, pauli_expansion
from .synthesis import synthesize, refine_circuit


class Block():
    """
    The Block Class.
    """

    def __init__ ( self, utry, link ):
        """
        Block Class Constructor

        Args:
            utry (np.array): Unitary

            link (tuple[int]): Link location
        """

        if not isinstance( utry, np.ndarray ):
            raise TypeError( "utry must be a np.ndarray." )

        if len( utry.shape ) != 2:
            raise TypeError( "utry must be a matrix." )

        if 2 ** len( link ) != utry.shape[0]:
            raise ValueError( "link and utry have different dimensions." )

        if 2 ** len( link ) != utry.shape[1]:
            raise ValueError( "link and utry have different dimensions." )

        if not np.allclose( utry @ utry.conj().T, np.identity( len( utry ) ) ):
            raise ValueError( "utry must be a unitary matrix." )

        if not np.allclose( utry.conj().T @ utry, np.identity( len( utry ) ) ):
            raise ValueError( "utry must be a unitary matrix." )

        self.utry = utry
        self.link = link
        self.num_qubits = len( link )

    def get_pauli_params ( self ):
        return pauli_expansion( unitary_log_no_i( self.utry ) )

    def __str__ ( self ):
        return str( self.link ) + ":" + str( self.utry )

    def __repr__ ( self ):
        return str( self.link )           \
               + ": [["                    \
               + str( self.utry[0][0] )   \
               + " ... "                  \
               + str( self.utry[-1][-1] ) \
               + "]]"
