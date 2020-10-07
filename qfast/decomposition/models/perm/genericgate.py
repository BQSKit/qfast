"""
This module implements the GenericGate Class.

A GenericGate is a gate with variable location and function.
This is done using permutation matrices.
"""


from copy import deepcopy

import numpy as np
import scipy as sp

from qfast import pauli
from qfast import perm
from qfast import utils
from qfast.decomposition.gatemodel import GateModel


class GenericGate ( GateModel ):

    def __init__ ( self, num_qubits, gate_size, locations ):
        """
        GenericGate Constructor

        Args:
            num_qubits (int): The number of qubits in the entire circuit

            gate_size (int): The number of qubits this gate acts on

            locations (list[tuple[int]]): The potential locations of this gate
        """

        super().__init__( num_qubits, gate_size )

        if not utils.is_valid_locations( locations, num_qubits, gate_size ):
            raise TypeError( "Specified locations is invalid." )

        self.locations = locations

        self.Hcoef = -1j / ( 2 ** self.num_qubits )
        self.paulis = pauli.get_norder_paulis( self.gate_size )
        self.sigmav = self.Hcoef * np.array( self.paulis )
        self.I = np.identity( 2 ** ( num_qubits - gate_size ) )
        self.perms = np.array( [ perm.calc_permutation_matrix( num_qubits, l )
                                 for l in self.locations ] )

        self.working_locations = deepcopy( locations )
        self.working_perms = np.copy( self.perms )

    def get_location ( self, x ):
        """Returns the gate's location."""
        idx = np.argmax( self.get_location_values( x ) )
        return self.working_locations[ idx ]

    def get_function_count ( self ):
        """Returns the number of function input parameters."""
        return 4 ** self.gate_size

    def get_location_count ( self ):
        """Returns the number of location input parameters."""
        return len( self.working_locations )

    def get_param_count ( self ):
        """Returns the number of the gate's input parameters."""
        return self.get_function_count() + self.get_location_count()

    def cannot_restrict ( self ):
        """Return true if the gate's location cannot be restricted."""
        return len( self.working_locations ) <= 1

    def restrict ( self, location ):
        """Restrict the gate's model by removing a potential location."""
        idx = self.working_locations.index( location )
        self.working_locations.pop( idx )
        self.working_perms = np.delete( self.working_perms, idx, 0 )

    def lift_restrictions ( self ):
        """Remove previous restrictions on the gate's model."""
        self.working_locations = deepcopy( self.locations )
        self.working_perms = np.copy( self.perms )

    def get_function_values ( self, x ):
        """Returns the function values."""
        return x[ : self.get_function_count() ]

    def get_location_values ( self, x ):
        """Returns the location values."""
        return x[ self.get_function_count() : ]

    def get_initial_input ( self ):
        """Produces a random vector of inputs."""
        ain = np.random.random( self.get_function_count() )
        # ain = [ np.pi ] * self.get_function_count()
        lin = [ 0 ] * self.get_location_count()
        return np.concatenate( [ ain, lin ] )

    def partition_input ( self, x ):
        """Splits the input vector into function and location values."""
        alpha = self.get_function_values( x )
        l = self.get_location_values( x )
        return alpha, l

    def get_gate_matrix ( self, x ):
        """Produces the matrix for this gate on its own."""
        sigma = pauli.get_norder_paulis( self.gate_size )
        sigma = self.Hcoef * sigma
        alpha = self.get_function_values( x )
        H = utils.dot_product( alpha, sigma )
        return sp.linalg.expm( H )

    def get_fixed_matrix ( self, x ):
        """Returns the fixed-location version of this gate's matrix."""
        alpha, l = self.partition_input( x )
        fixed_location = np.argmax( l )
        H = utils.dot_product( alpha, self.sigmav )
        U = sp.linalg.expm( H )
        P = self.working_perms[ fixed_location ]
        return P @ np.kron( U, self.I ) @ P.T

    def get_matrix ( self, x ):
        """Produces the circuit matrix for this gate."""
        alpha, l = self.partition_input( x )
        l = utils.softmax( l, 10 )

        H = utils.dot_product( alpha, self.sigmav )
        U = sp.linalg.expm( H )
        P = utils.dot_product( l, self.working_perms )
        return P @ np.kron( U, self.I ) @ P.T

    def get_matrix_and_derivatives ( self, x ):
        """Produces the circuit matrix and partials for this gate."""
        alpha, l = self.partition_input( x )
        l = utils.softmax( l, 10 )

        H = utils.dot_product( alpha, self.sigmav )
        P = utils.dot_product( l, self.working_perms )
        U = np.kron( sp.linalg.expm( H ), self.I )
        PU = P @ U
        UP = U @ P.T
        PUP =  PU @ P.T

        _, dav = utils.dexpmv( H, self.sigmav )
        dav = np.kron( dav, self.I )
        dav = P @ dav @ P.T
        dlv = self.working_perms @ UP + PU @ self.working_perms.transpose( ( 0, 2, 1 ) ) - 2*PUP
        dlv = np.array( [ x*y for x, y in zip( 10*l, dlv ) ] )
        return PUP, np.concatenate( [ dav, dlv ] )

