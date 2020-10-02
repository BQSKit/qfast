"""
This module implements the Gate Class.

A gate is a unitary operation applied to a set of qubits.
"""

import numpy as np

from qfast import utils

class Gate():
    """The Gate Class."""

    def __init__ ( self, utry, location ):
        """
        Gate Class Constructor

        Args:
            utry (np.ndarray): The gate's unitary operation.

            location (tuple[int]): The set of qubits the gate acts on.

        Raises:
            TypeError: If unitary or location are invalid.
        """

        if not utils.is_unitary( utry, tol = 1e-14 ):
            raise TypeError( "Invalid unitary." )

        self.utry = utry
        self.num_qubits = utils.get_num_qubits( self.utry )

        if not utils.is_valid_location( location ):
            raise TypeError( "Invalid location." )

        if len( location ) != self.num_qubits:
            raise ValueError( "Invalid size of location." )

        self.location = location

    def __str__ ( self ):
        """Gets the gate's string representation."""

        return str( self.location ) + ":" + str( self.utry )

    def __repr__ ( self ):
        """Gets a simple gate string representation."""

        return str( self.location )       \
               + ": [["                   \
               + str( self.utry[0][0] )   \
               + " ... "                  \
               + str( self.utry[-1][-1] ) \
               + "]]"

