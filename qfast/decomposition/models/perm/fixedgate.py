import numpy as np
import scipy.linalg as la

from qfast import pauli
from qfast import perm
from qfast import utils

class FixedGate():

    def __init__ ( self, num_qubits, gate_size, location ):
        assert( len( location ) == gate_size )

        self.num_qubits = num_qubits
        self.gate_size = gate_size
        self.location = location

        self.Hcoef = -1j / ( 2 ** self.num_qubits )
        self.paulis = pauli.get_norder_paulis( self.gate_size )
        self.sigmav = self.Hcoef * np.array( self.paulis )
        self.I = np.identity( 2 ** ( num_qubits - gate_size ) )
        self.perm_matrix = perm.calc_permutation_matrix( num_qubits, location )

    def get_location ( self ):
        return self.location

    def get_alpha_count ( self ):
        return 4 ** self.gate_size

    def get_location_count ( self ):
        return 1

    def get_param_count ( self ):
        return self.get_alpha_count()

    def get_initial_input ( self ):
        return [ np.pi ] * (4 ** self.gate_size)

    def get_matrix ( self, x ):
        H = utils.dot_product( x, self.sigmav )
        U = la.expm( H )
        P = self.perm_matrix
        return P @ np.kron( U, self.I ) @ P.T

    def get_actual_matrix ( self, x ):
        sigma = pauli.get_norder_paulis( self.gate_size )
        sigma = self.Hcoef * sigma
        H = utils.dot_product( x, sigma )
        return la.expm( H )

    def get_matrix_and_derivatives ( self, x ):
        H = utils.dot_product( x, self.sigmav )
        P = self.perm_matrix
        U = np.kron( la.expm( H ), self.I )
        PUP = P @ U @ P.T

        _, dav = utils.dexpmv( H, self.sigmav )
        dav = np.kron( dav, self.I )
        dav = P @ dav @ P.T
        return PUP, dav

