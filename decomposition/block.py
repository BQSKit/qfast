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
        self.size = len( link )

    def decompose ( self, **kwargs ):

        if self.size <= 2:
            return self

        params["start_depth" ] = 1
        params["depth_step" ] = 1
        params["exploration_distance" ] = 0.01
        params["exploration_learning_rate" ] = 0.01
        params["refinement_distance" ] = 1e-7
        params["refinement_learning_rate" ] = 1e-6
        params.update( kwargs )

        gate_size = self.get_decomposition_size()

        # Explore
        fun_vals, loc_vals = exploration( self.utry, self.size, gate_size,
                                          params["start_depth"],
                                          params["depth_step"],
                                          params["exploration_distance"],
                                          params["exploration_learning_rate"] )

        loc_fixed = fix_locations( loc_vals, gate_size )

        # if verbosity >= 1:
        #     print( "Found circuit in %f seconds" % (end - start) )

        # Refine
        fun_vals = refinement( self.utry, self.size, gate_size,
                               fun_vals, loc_vals,
                               params["refinement_distance"],
                               params["refinement_learning_rate"] )

        # if verbosity >= 1:
            # print( "Refined circuit in %f seconds" % (end - start) )

        # Piece Together
        block_list = []

        for loc, fun_params in zip( loc_fixed, fun_vals ):
            gate_utry = get_unitary_from_pauli_coefs( params )
            mapped_link = tuple( [ self.link[i] for i in link ] )
            block_list.append( Block( gate_utry, mapped_link ) )

        return block_list

    def get_decomposition_size ( self ):
        return int( np.ceil( self.size / 2 ) )

    def fix_locations ( self, loc_vals, gate_size ):
        # This is bad programing since it's duplicated code
        # from different parts of the program and is more or less a hack
        # TODO: write a HardwareModel class that models a target
        # hardware architecture and can be queried for this and
        # other information.
        loc_fixed = []
        topology = list( it.combinations( range( self.size ), gate_size ) )

        for i, loc_val in enumerate( loc_vals ):
            loc_idx = np.argmax( loc_val )
            parity = (i + 1) % 2
            if parity == 0:
                loc_fixed.append( topology[:len(topology)//2][loc_idx] )
            elif parity == 1:
                loc_fixed.append( topology[len(topology)//2:][loc_idx] )

        return loc_fixed

    def get_pauli_params ( self ):
        return pauli_expansion( unitary_log_no_i( self.utry ) )

    def __str__ ( self ):
        return str( self.link ) + ":" + str( self.utry )

    def __repr__ ( self ):
        return str( self.link )           \
               + ": ["                    \
               + str( self.utry[0][0] )   \
               + " ... "                  \
               + str( self.utry[-1][-1] ) \
               + "]"
