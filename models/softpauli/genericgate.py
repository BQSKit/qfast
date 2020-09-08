from copy import deepcopy
import scipy.linalg as la
import itertools as it
import numpy as np
from qfast.pauli import get_norder_paulis, get_pauli_n_qubit_projection

from qfast.utils import is_unitary, is_skew_hermitian, is_hermitian, softmax, dexpmv, dot_product

class GenericGate():

    def __init__ ( self, num_qubits, gate_size, locations ):
        self.Hcoef  = -1j / ( 2 ** num_qubits )
        self.locations = locations
        self.paulis = [ get_pauli_n_qubit_projection( num_qubits, location )
                        for location in locations ]
        self.sigmav = self.Hcoef * np.array( self.paulis )

        self.working_locations = deepcopy( locations )
        self.working_sigmav = np.copy( self.sigmav )

    def get_chosen_location ( self, x ):
        #print( self.working_locations )
        #print( x[ -self.get_l_count() : ] )
        return self.working_locations[ np.argmax( x[ -self.get_l_count() : ] ) ]

    def cannot_restrict ( self ):
        return len( self.working_locations ) <= 1

    def restrict ( self, location ):
        idx = self.working_locations.index( location )
        self.working_locations.pop( idx )
        # print( idx )
        # print( self.working_sigmav.shape )
        self.working_sigmav = np.delete( self.working_sigmav, idx, 0 )
        # print( self.working_sigmav.shape )

    def lift_restrictions ( self ):
        self.working_locations = deepcopy( self.locations )
        self.working_sigmav = np.copy( self.sigmav )
        #print( self.working_sigmav.shape )
        #print( self.locations )

    def convert_input_to_fixed ( self, x ):
        idx = np.argmax( x[ -self.get_l_count() : ] )
        return x[ idx*self.working_sigmav.shape[1] : (idx + 1) * self.working_sigmav.shape[1] ]

    def get_alpha_count ( self ):
        return self.working_sigmav.shape[0]*self.working_sigmav.shape[1]

    def get_l_count ( self ):
        return self.working_sigmav.shape[0]

    def get_param_count ( self ):
        return self.get_alpha_count() + self.get_l_count()

    def get_initial_input ( self ):
        a0 = np.random.random( self.get_alpha_count() )
        l0 = [ 0 ] * self.working_sigmav.shape[0]
        return np.concatenate( ( a0, l0 ) )

    def get_lower_bounds ( self ):
        a0 = [ -8*np.pi ] * self.working_sigmav.shape[0]*self.working_sigmav.shape[1]
        l0 = [ -10 ] * self.working_sigmav.shape[0]
        return np.concatenate( ( a0, l0 ) )

    def get_upper_bounds ( self ):
        a0 = [ 8*np.pi ] * self.working_sigmav.shape[0]*self.working_sigmav.shape[1]
        l0 = [ 10 ] * self.working_sigmav.shape[0]
        return np.concatenate( ( a0, l0 ) )

    def get_fixed_matrix ( self, x ):
        alpha = x[ : self.get_alpha_count() ]
        l = x[ self.get_alpha_count() : ]

        fixed_location = np.argmax( l )
        H = dot_product( alpha[ fixed_location * self.working_sigmav.shape[1]: (fixed_location + 1) * self.working_sigmav.shape[1] ], self.working_sigmav[ fixed_location ] )
        return la.expm( H )

    def get_matrix ( self, x ):
        alpha = x[ : self.get_alpha_count() ]
        l = x[ self.get_alpha_count() : ]

        Hv = []

        for i in range( self.working_sigmav.shape[0] ):
            Hv.append( dot_product( alpha[self.working_sigmav.shape[1] * i : self.working_sigmav.shape[1] * (i + 1)], self.working_sigmav[i] ) )
        beta = 20
        l = softmax( l, beta )
        H = dot_product( l, Hv )
        return la.expm( H )

    def get_derivative ( self, x ):
        alpha = x[ : self.get_alpha_count() ]
        l = x[ self.get_alpha_count() : ]

        Hv = []

        for i in range( self.working_sigmav.shape[0] ):
            Hv.append( dot_product( alpha[self.working_sigmav.shape[1] * i : self.working_sigmav.shape[1] * (i + 1)], self.working_sigmav[i] ) )
        beta = 20
        l = softmax( l, beta )

        H = dot_product( l, Hv )

        # Partials of H with respect to alpha variables
        alpha_der = np.array( [ l[i] * self.working_sigmav[i]
                                for i in range( self.working_sigmav.shape[0] ) ] )
        alpha_der = alpha_der.reshape( ( -1, alpha_der.shape[-2], alpha_der.shape[-1] ) )

        L = np.tile( l, ( self.working_sigmav.shape[0], 1 ) )
        L = np.identity( self.working_sigmav.shape[0] ) - L
        L = beta * ( np.diag( l ) @ L )

        l_der = np.array( [ dot_product( Lr, Hv ) for Lr in L ] )

        _, dav = dexpmv( H, alpha_der )
        _, dlv = dexpmv( H, l_der )

        return np.concatenate( [ dav, dlv ] )

    def get_matrix_and_derivatives ( self, x ):
        alpha = x[ : self.get_alpha_count() ]
        l = x[ self.get_alpha_count() : ]

        Hv = []

        for i in range( self.working_sigmav.shape[0] ):
            Hv.append( dot_product( alpha[self.working_sigmav.shape[1] * i : self.working_sigmav.shape[1] * (i + 1)], self.working_sigmav[i] ) )
        beta = 20
        l = softmax( l, beta )

        H = dot_product( l, Hv )

        # Partials of H with respect to alpha variables
        alpha_der = np.array( [ l[i] * self.working_sigmav[i]
                                for i in range( self.working_sigmav.shape[0] ) ] )
        alpha_der = alpha_der.reshape( ( -1, alpha_der.shape[-2], alpha_der.shape[-1] ) )

        L = np.tile( l, ( self.working_sigmav.shape[0], 1 ) )
        L = np.identity( self.working_sigmav.shape[0] ) - L
        L = beta * ( np.diag( l ) @ L )

        l_der = np.array( [ dot_product( Lr, Hv ) for Lr in L ] )

        _, dav = dexpmv( H, alpha_der )
        _, dlv = dexpmv( H, l_der )

        return la.expm( H ), np.concatenate( [ dav, dlv ] )

