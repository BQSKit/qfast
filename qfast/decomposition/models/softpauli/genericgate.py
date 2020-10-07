"""
This module implements the GenericGate Class.

A GenericGate is a gate with variable location and function.
This is done using the softmax of all groups of pauli matrices.
"""


from copy import deepcopy

import numpy as np
import scipy as sp

from qfast import pauli
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

        self.Hcoef  = -1j / ( 2 ** num_qubits )
        self.locations = locations
        self.paulis = [ pauli.get_pauli_n_qubit_projection( num_qubits, location )
                        for location in locations ]
        self.sigmav = self.Hcoef * np.array( self.paulis )

        self.working_locations = deepcopy( locations )
        self.working_sigmav = np.copy( self.sigmav )

    def get_location ( self, x ):
        """Returns the gate's location."""
        idx = np.argmax( self.get_location_values( x ) )
        return self.working_locations[ idx ]

    def get_function_count ( self ):
        """Returns the number of function input parameters."""
        return (4 ** self.gate_size) * self.get_location_count()

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
        self.working_sigmav = np.delete( self.working_sigmav, idx, 0 )

    def lift_restrictions ( self ):
        """Remove previous restrictions on the gate's model."""
        self.working_locations = deepcopy( self.locations )
        self.working_sigmav = np.copy( self.sigmav )

    def get_function_values ( self, x, only_max = False ):
        """Returns the function values."""

        if only_max:
            idx = np.argmax( self.get_location_values( x ) )
            stride = 4 ** self.gate_size
            return x[ idx * stride : (idx + 1) * stride ]

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
        a = self.get_function_values( x )
        l = self.get_location_values( x )
        return a, l

    def get_gate_matrix ( self, x ):
        """Produces the matrix for this gate on its own."""
        sigma = pauli.get_norder_paulis( self.gate_size )
        sigma = self.Hcoef * sigma
        alpha = self.get_function_values( x, True )
        H = utils.dot_product( alpha, sigma )
        return sp.linalg.expm( H )

    def get_fixed_matrix ( self, x ):
        """Returns the fixed-location version of this gate's matrix."""
        alpha, l = self.partition_input( x )
        fixed_location = np.argmax( l )
        H = utils.dot_product( self.get_function_values( x, True ),
                               self.working_sigmav[ fixed_location ] )
        return sp.linalg.expm( H )

    def get_matrix ( self, x ):
        """Produces the circuit matrix for this gate."""
        alpha, l = self.partition_input( x )
        l = utils.softmax( l, 10 )

        Hv = []

        stride = self.working_sigmav.shape[1]
        for i in range( self.working_sigmav.shape[0] ):
            Hv.append( utils.dot_product( alpha[ i * stride : (i+1) * stride ],
                                          self.working_sigmav[i] ) )
        H = utils.dot_product( l, Hv )
        return sp.linalg.expm( H )

    def get_matrix_and_derivatives ( self, x ):
        """Produces the circuit matrix and partials for this gate."""
        alpha, l = self.partition_input( x )
        l = utils.softmax( l, 10 )

        Hv = []

        stride = self.working_sigmav.shape[1]
        for i in range( self.working_sigmav.shape[0] ):
            Hv.append( utils.dot_product( alpha[ i * stride : (i+1) * stride ],
                                          self.working_sigmav[i] ) )

        H = utils.dot_product( l, Hv )

        # Partials of H with respect to function variables
        alpha_der = np.array( [ l[i] * self.working_sigmav[i]
                                for i in range( self.get_location_count() ) ] )
        alpha_der = alpha_der.reshape( ( -1, alpha_der.shape[-2],
                                             alpha_der.shape[-1] ) )

        # Partials of H with respect to location variables
        L = np.tile( l, ( self.working_sigmav.shape[0], 1 ) )
        L = np.identity( self.working_sigmav.shape[0] ) - L
        L = 10 * ( np.diag( l ) @ L )

        l_der = np.array( [ utils.dot_product( Lr, Hv ) for Lr in L ] )

        _, dav = utils.dexpmv( H, alpha_der )
        _, dlv = utils.dexpmv( H, l_der )

        return sp.linalg.expm( H ), np.concatenate( [ dav, dlv ] )

