"""
This module implements the Circuit Class.

A Circuit has blocks.
"""

import numpy as np
from block import Block
from synthesis import refine_circuit
from pauli import get_unitary_from_pauli_coefs

from qiskit import *


class Circuit():
    """
    The Circuit Class.
    """

    def __init__ ( self, utry ):
        """
        Circuit Class Constructor.

        Args:
            utry (np.array): creates a circuit with a single block
        """

        self.utry = utry
        self.num_qubits = int( np.log2( len( utry ) ) )
        self.blocks = [ Block( self.utry, list( range( self.num_qubits ) ) ) ]

    def decompose ( self, native_block_size, **kwargs ):
        """
        Decomposition breaks down the circuit into blocks of at most
        native_block_size size.

        Args:
            native_block_size (int): target block size
        """

        while any( [ block.size > native_block_size for block in self.blocks ] ):

            new_block_list = []

            for block in self.blocks:
               # if verbosity >= 1:
               #     print( "Synthesizing block: ", block.__repr__() )

                if block.size <= native_block_size:
                    new_block_list.append( block )
                else:
                    new_block_list += block.decompose( kwargs )

            self.blocks = new_block_list

        # Final Refinement
        circ_as_paulis = [ ( b.link, b.get_pauli_params() )
                           for b in self.blocks ]
        circ_as_paulis = refine_circuit( self.utry, circ_as_paulis )

        # Piece Together
        block_list = []

        for link, params in circ_as_paulis:
            gate_utry = get_unitary_from_pauli_coefs( params )
            mapped_link = link
            block_list.append( Block( gate_utry, mapped_link ) )

        self.blocks = block_list

    def dump_blocks ( self ):
        for i, block in enumerate( self.blocks ):
            linkname = str( block.link ).replace( ", ", "_" ).replace("(", "").replace(")", "")
            filename = "%d_%s.unitary" % ( i, linkname )
            filename = os.path.join( args.output, filename )
            np.savetxt( filename, block.utry )

