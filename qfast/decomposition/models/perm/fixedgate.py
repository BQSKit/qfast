"""
This module implements the FixedGate Class.

A FixedGate is a gate with fixed location but variable function.
This is done using permutation matrices.
"""


import numpy as np
import scipy as sp

from qfast import pauli
from qfast import perm
from qfast import utils
from qfast.decomposition.gatemodel import GateModel


class FixedGate ( GateModel ):

    def __init__ ( self, num_qubits, gate_size, location ):
        """
        FixedGate Constructor

        Args:
            num_qubits (int): The number of qubits in the entire circuit

            gate_size (int): The number of qubits this gate acts on

            location (tuple[int]): The qubits this gate acts on
        """

        super().__init__( num_qubits, gate_size )

        if not utils.is_valid_location( location, num_qubits ):
            raise TypeError( "Specified location is invalid." )

        if len( location ) != gate_size:
            raise ValueError( "Location does not match gate size." )

        self.location = location

        self.Hcoef = -1j / ( 2 ** self.num_qubits )
        self.paulis = pauli.get_norder_paulis( self.gate_size )
        self.sigmav = self.Hcoef * np.array( self.paulis )
        self.I = np.identity( 2 ** ( num_qubits - gate_size ) )
        self.perm_matrix = perm.calc_permutation_matrix( num_qubits, location )

    def get_location ( self, x ):
        """Returns the gate's location."""
        return self.location

    def get_param_count ( self ):
        """Returns the number of the gate's input parameters."""
        return 4 ** self.gate_size

    def get_matrix ( self, x ):
        """Produces the circuit matrix for this gate."""
        H = utils.dot_product( x, self.sigmav )
        U = sp.linalg.expm( H )
        P = self.perm_matrix
        return P @ np.kron( U, self.I ) @ P.T

    def get_gate_matrix ( self, x ):
        """Produces the matrix for this gate on its own."""
        sigma = pauli.get_norder_paulis( self.gate_size )
        sigma = self.Hcoef * sigma
        H = utils.dot_product( x, sigma )
        return sp.linalg.expm( H )

    def get_matrix_and_derivatives ( self, x ):
        """Produces the circuit matrix and partials for this gate."""
        H = utils.dot_product( x, self.sigmav )
        P = self.perm_matrix
        U = np.kron( sp.linalg.expm( H ), self.I )
        PUP = P @ U @ P.T

        _, dav = utils.dexpmv( H, self.sigmav )
        dav = np.kron( dav, self.I )
        dav = P @ dav @ P.T
        return PUP, dav

