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

    def __init__ ( self, utry, kernel = "kak" ):
        """
        Circuit Class Constructor

        Args:
            utry (np.array): Unitary

            kernel (str): Either "kak" or "uq":
                          "kak" uses QFAST to break down blocks until they have
                          size 2; then uses the KAK decomposition to convert to
                          native gates

                          "uq" uses QFAST to break down blocks until they have
                          size 3; then uses the UniversalQ Compiler to convert
                          to native gates
        """

        self.utry = utry
        self.num_qubits = int( np.log2( len( utry ) ) )
        self.kernel = kernel

        self.blocks = [ Block( self.utry, list( range( self.num_qubits ) ) ) ]

    def synthesize ( self, verbosity = 0 ):
        if self.kernel == "kak":
            kernel_size = 2
        elif self.kernel == "uq":
            kernel_size = 3

        while any( [ block.size > kernel_size for block in self.blocks ] ):

            new_block_list = []

            for block in self.blocks:
                if verbosity >= 1:
                    print( "Synthesizing block: ", block.__repr__() )

                if block.size <= kernel_size:
                    new_block_list.append( block )
                else:
                    new_block_list += block.synthesize( verbosity )

            self.blocks = new_block_list

        # Final Refinement
        circ_as_paulis = [ ( b.link, b.get_pauli_params() )
                           for b in self.blocks ]
        circ_as_paulis = refine_circuit( self.utry, circ_as_paulis, verbosity )

        # Piece Together
        block_list = []

        for link, params in circ_as_paulis:
            gate_utry = get_unitary_from_pauli_coefs( params )
            mapped_link = link
            block_list.append( Block( gate_utry, mapped_link ) )

        self.blocks = block_list

        if self.kernel == "kak":
            return self.qiskit_kak_decomp()
        elif self.kernel == "uq":
            return self.universalq_decomp()

    def qiskit_kak_decomp ( self ):
        assert( all( [ block.size <= 2 for block in self.blocks ] ) )

        circ = QuantumCircuit( self.num_qubits )

        for block in self.blocks:
            assert( len( block.link ) == 2 )
            circ.unitary( block.utry, [ block.link[1], block.link[0] ] )

        circ = qiskit.compiler.transpile( circ, basis_gates = ['u3', 'cx'],
                                          optimization_level = 3 )
        circ = qiskit.compiler.transpile( circ, basis_gates = ['u3', 'cx'],
                                          optimization_level = 3 )
        circ = qiskit.compiler.transpile( circ, basis_gates = ['u3', 'cx'],
                                          optimization_level = 3 )

        return circ.qasm()

    def universalq_decomp ( self ):
        pass