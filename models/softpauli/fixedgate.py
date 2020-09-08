import itertools

import numpy as np
import scipy as sp

from qfast.pauli import get_pauli_n_qubit_projection, get_norder_paulis
from qfast.utils import dexpmv, dot_product

class FixedGate():

    def __init__ ( self, num_qubits, gate_size, location ):
        self.Hcoef  = -1j / ( 2 ** num_qubits )
        self.location = location
        self.gate_size = gate_size
        self.num_qubits = num_qubits
        self.sigmav = get_pauli_n_qubit_projection( num_qubits, location )
        self.sigmav = self.Hcoef * self.sigmav

    def get_location ( self ):
        return self.location

    def get_param_count ( self ):
        return self.sigmav.shape[0]

    def get_initial_input ( self ):
        return np.random.random( self.sigmav.shape[0] )

    def get_matrix ( self, x ):
        H = dot_product( x, self.sigmav )
        return sp.linalg.expm( H )

    def get_actual_matrix ( self, x ):
        sigma = get_norder_paulis( self.gate_size )
        sigma = self.Hcoef * sigma
        H = dot_product( x, sigma )
        return sp.linalg.expm( H )

    def get_derivative ( self, x ):
        H = dot_product( x, self.sigmav )
        return dexpmv( H, self.sigmav )[1]

    def get_matrix_and_derivatives ( self, x ):
        H = dot_product( x, self.sigmav )
        return dexpmv( H, self.sigmav )

