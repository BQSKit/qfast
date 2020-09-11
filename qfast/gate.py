"""
This module implements the Gate Class.

A gate is a unitary operation applied to a set of qubits.
"""

import numpy as np

from .utils import is_unitary

class Gate():
    """The Gate Class."""

    def __init__ ( self, utry, loc ):
        """
        Gate Class Constructor

        Args:
            utry (np.ndarray): Unitary

            location (tuple[int]): Gate location (set of qubits)
        """

        if not is_unitary( utry, tol = 1e-15 ):
            raise TypeError( "utry must be a valid unitary matrix." )

        if not isinstance( loc, tuple ):
            raise TypeError( "loc must be a location." )

        if not all( isinstance( q, int ) for q in loc ):
            raise TypeError( "loc must be a location." )

        if not len( loc ) == len( set( loc ) ):
            raise TypeError( "loc must be a valid location." )

        if 2 ** len( loc ) != utry.shape[0]:
            raise ValueError( "loc and utry have incompatible dimensions." )

        self.utry = utry
        self.loc  = loc
        self.num_qubits = len( loc )

    def get_utry ( self ):
        """
        Gets the gate's unitary.

        Returns:
            (np.ndarray): Gate's unitary
        """

        return self.utry

    def get_location ( self ):
        """
        Gets the gate's location.

        Returns:
            (Tuple[int]): Gate's location
        """

        return self.loc

    def get_size ( self ):
        """
        Gets the gate's size in qubits.

        Returns:
            (int): Gate's size
        """

        return self.num_qubits

    def __str__ ( self ):
        """
        Gets the gate's string representation.

        Returns:
            (str): Gate's string representation
        """

        return str( self.loc ) + ":" + str( self.utry )

    def __repr__ ( self ):
        """
        Gets a simple gate string representation.

        Returns:
            (str): Gate's simple string representation
        """

        return str( self.loc )           \
               + ": [["                    \
               + str( self.utry[0][0] )   \
               + " ... "                  \
               + str( self.utry[-1][-1] ) \
               + "]]"

