"""
This module implements the Circuit Class.

A Circuit holds many blocks and implements methods for synthesis.
"""

import os

import numpy as np

from .block import Block
from .decomposition import decomposition, refinement, convert_to_block_list


class Circuit():
    """
    The Circuit Class.
    """

    def __init__ ( self, utry ):
        """
        Circuit Class Constructor.

        Args:
            utry (np.ndarray): Creates a circuit with a single block
        """

        if not isinstance( utry, np.ndarray ):
            raise TypeError( "utry must be a np.ndarray." )

        if len( utry.shape ) != 2:
            raise TypeError( "utry must be a matrix." )

        if utry.shape[0] != utry.shape[1]:
            raise TypeError( "utry must be a square matrix." )

        if not np.allclose( utry @ utry.conj().T, np.identity( len( utry ) ) ):
            raise ValueError( "utry must be a unitary matrix." )

        if not np.allclose( utry.conj().T @ utry, np.identity( len( utry ) ) ):
            raise ValueError( "utry must be a unitary matrix." )

        self.utry = utry
        self.num_qubits = int( np.log2( len( utry ) ) )
        self.blocks = [ Block( self.utry, tuple( range( self.num_qubits ) ) ) ]

    def hierarchically_decompose ( self, native_block_size, **kwargs ):
        """
        Hierarchically decompose a circuit into blocks of size at most
        native_block_size.

        Args:
            native_block_size (int): target block size

        Keyword Args:
            See decomposition in decomposition.py
        """

        # Decompose circuit
        while any( [ block.num_qubits > native_block_size
                     for block in self.blocks ] ):

            new_block_list = []

            for block in self.blocks:

                if block.num_qubits <= native_block_size:
                    new_block_list.append( block )
                else:
                    kwargs["native_block_size"] = native_block_size
                    new_block_list += decomposition( block, **kwargs )

            self.blocks = new_block_list

        # Final Refinement
        fun_vals, loc_fixed = self.get_fun_loc_vals()

        params = { "refinement_distance": 1e-7,
                   "refinement_learning_rate": 1e-6 }

        if "refinement_distance" in kwargs:
            params["refinement_distance"] = \
                kwargs["refinement_distance"]

        if "refinement_learning_rate" in kwargs:
            params["refinement_learning_rate"] = \
                kwargs["refinement_learning_rate"]

        fun_vals = refinement( self.utry, self.num_qubits, len( loc_fixed[0] ),
                               fun_vals, loc_fixed,
                               params["refinement_distance"],
                               params["refinement_learning_rate"] )

        self.blocks = convert_to_block_list( list( range( self.num_qubits ) ),
                                             fun_vals, loc_fixed )

    def get_fun_loc_vals ( self ):
        """
        Converts the circuit's block list into location and function
        value lists.

        Returns:
            (Tuple[List[List[float]], List[Tuple[int]]):
                The circuit block's function values and locations
        """

        return ( [ b.get_fun_vals() for b in self.blocks ],
                 [ b.get_location() for b in self.blocks ] )

    def get_locations ( self ):
        """
        Gets the locations of all the blocks in the circuit.

        Returns:
            (List[Tuple[int]]): The circuit block's locations
        """

        return [ b.get_location() for b in self.blocks ]

    def dump_blocks ( self, directory ):
        """
        Dumps the circuit's blocks into a directory.

        Args:
            directory (str): The directory where blocks will be dumped.
        """

        if not os.path.isdir( directory ):
            raise ValueError( "Invalid directory: %s" % directory )

        for i, block in enumerate( self.blocks ):
            linkname = str( block.get_location() ).replace( ", ", "_" )
            linkname = linkname.replace("(", "").replace(")", "")
            filename = "%d_%s.unitary" % ( i, linkname )
            filename = os.path.join( directory, filename )
            np.savetxt( filename, block.get_utry() )
