"""
This module implements the Block Class.

A block is a unitary operation applied to a set of qubits.
"""

import numpy as np

from .pauli import unitary_log_no_i, pauli_expansion


class Block():
    """
    The Block Class.
    """

    def __init__ ( self, utry, loc ):
        """
        Block Class Constructor

        Args:
            utry (np.ndarray): Unitary

            location (tuple[int]): Block location (set of qubits)
        """

        if not isinstance( utry, np.ndarray ):
            raise TypeError( "utry must be a np.ndarray." )

        if len( utry.shape ) != 2:
            raise TypeError( "utry must be a matrix." )

        if not isinstance( loc, tuple ):
            raise TypeError( "loc must be a location." )

        if not all( isinstance( q, int ) for q in loc ):
            raise TypeError( "loc must be a location." )

        if not len( loc ) == len( set( loc ) ):
            raise TypeError( "loc must be a valid location." )

        if 2 ** len( loc ) != utry.shape[0]:
            raise ValueError( "loc and utry have incompatible dimensions." )

        if 2 ** len( loc ) != utry.shape[1]:
            raise ValueError( "loc and utry have incompatible dimensions." )

        if not np.allclose( utry @ utry.conj().T, np.identity( len( utry ) ) ):
            raise ValueError( "utry must be a unitary matrix." )

        if not np.allclose( utry.conj().T @ utry, np.identity( len( utry ) ) ):
            raise ValueError( "utry must be a unitary matrix." )

        self.utry = utry
        self.loc  = loc
        self.num_qubits = len( loc )

    def get_utry ( self ):
        """
        Gets the block's unitary.

        Returns:
            (np.ndarray): Block's unitary
        """

        return self.utry

    def get_fun_vals ( self ):
        """
        Gets the block's function parameters.

        Returns:
            (List[float]): Block's function parameters
        """

        return pauli_expansion( unitary_log_no_i( self.utry ) )

    def get_location ( self ):
        """
        Gets the block's location.

        Returns:
            (Tuple[int]): Block's location
        """

        return self.loc

    def get_num_qubits ( self ):
        """
        Gets the block's size in qubits.

        Returns:
            (int): Block's size
        """

        return self.num_qubits

    def __str__ ( self ):
        """
        Gets the block's string representation.

        Returns:
            (str): Block's string representation
        """

        return str( self.loc ) + ":" + str( self.utry )

    def __repr__ ( self ):
        """
        Gets a simple block string representation.

        Returns:
            (str): Block's simple string representation
        """

        return str( self.loc )           \
               + ": [["                    \
               + str( self.utry[0][0] )   \
               + " ... "                  \
               + str( self.utry[-1][-1] ) \
               + "]]"
