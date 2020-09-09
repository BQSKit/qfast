from copy import deepcopy

import numpy as np
import scipy.linalg as la

from qfast import pauli
from qfast import perm
from qfast import utils

class GenericGate():

    def __init__ ( self, num_qubits, gate_size, locations ):
        assert( all( [ len( l ) == gate_size for l in locations ] ) )

        self.num_qubits = num_qubits
        self.gate_size = gate_size
        self.locations = locations

        self.Hcoef = -1j / ( 2 ** self.num_qubits )
        self.paulis = pauli.get_norder_paulis( self.gate_size )
        self.sigmav = self.Hcoef * np.array( self.paulis )
        self.I = np.identity( 2 ** ( num_qubits - gate_size ) )
        self.perm_matrices = np.array( [ perm.calc_permutation_matrix( num_qubits, location )
                                         for location in self.locations ] )

        self.working_locations = deepcopy( locations )
        self.working_perm = np.copy( self.perm_matrices )

    def get_chosen_location ( self, x ):
        return self.working_locations[ np.argmax( x[ -self.get_location_count() : ] ) ]

    def cannot_restrict ( self ):
        return len( self.working_locations ) <= 1

    def restrict ( self, location ):
        idx = self.working_locations.index( location )
        self.working_locations.pop( idx )
        # print( idx )
        # print( self.working_sigmav.shape )
        self.working_perm = np.delete( self.working_perm, idx, 0 )
        # print( self.working_sigmav.shape )

    def lift_restrictions ( self ):
        self.working_locations = deepcopy( self.locations )
        self.working_perm = np.copy( self.perm_matrices )
        #print( self.working_sigmav.shape )
        #print( self.locations )

    def convert_input_to_fixed ( self, x ):
        return x[ : -self.get_location_count() ]

    def get_alpha_count ( self ):
        return 4 ** self.gate_size

    def get_location_count ( self ):
        return len( self.working_locations )

    def get_param_count ( self ):
        return self.get_alpha_count() + self.get_location_count()

    def get_initial_input ( self ):
        ain = [ np.pi ] * (4 ** self.gate_size)
        lin = [ 0 ] * len( self.working_locations )
        return np.concatenate( [ ain, lin ] )

    def partition_input ( self, x ):
        alpha = x[ : self.get_alpha_count() ]
        l = x[ self.get_alpha_count() : ]
        return alpha, l

    def get_fixed_matrix ( self, x ):
        alpha, l = self.partition_input( x )
        fixed_location = np.argmax( l )

        H = utils.dot_product( alpha, self.sigmav )
        U = la.expm( H )
        P = self.working_perm[ fixed_location ]
        return P @ np.kron( U, self.I ) @ P.T


    def get_matrix ( self, x ):
        alpha, l = self.partition_input( x )
        l = utils.softmax( l, 10 )

        H = utils.dot_product( alpha, self.sigmav )
        U = la.expm( H )
        P = utils.dot_product( l, self.working_perm )
        return P @ np.kron( U, self.I ) @ P.T

    def get_matrix_and_derivatives ( self, x ):
        alpha, l = self.partition_input( x )
        l = utils.softmax( l, 10 )

        H = utils.dot_product( alpha, self.sigmav )
        P = utils.dot_product( l, self.working_perm )
        U = np.kron( la.expm( H ), self.I )
        PU = P @ U
        UP = U @ P.T
        PUP =  PU @ P.T

        _, dav = utils.dexpmv( H, self.sigmav )
        dav = np.kron( dav, self.I )
        dav = P @ dav @ P.T
        dlv = self.working_perm @ UP + PU @ self.working_perm.transpose( ( 0, 2, 1 ) ) - 2*PUP
        dlv = np.array( [ x*y for x, y in zip( 10*l, dlv ) ] )
        return PUP, np.concatenate( [ dav, dlv ] )

