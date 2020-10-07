"""
This module implements the GateModel abstract class.

A GateModel is a parameterized gate.
"""

import abc
import numpy as np
import scipy as sp

from qfast import pauli
from qfast import perm
from qfast import utils

class GateModel( abc.ABC ):

    def __init__ ( self, num_qubits, gate_size ):
        """
        GateModel Constructor

        Args:
            num_qubits (int): The number of qubits in the entire circuit

            gate_size (int): The number of qubits this gate acts on
        """

        if num_qubits <= 0:
            raise ValueError( "Must have positive number of qubits." )

        if gate_size <= 0:
            raise ValueError( "Gate size must be a positive integer." )

        if gate_size > num_qubits:
            raise ValueError( "Gate size must be less than total qubits." )

        self.num_qubits = num_qubits
        self.gate_size = gate_size

    def get_initial_input ( self ):
        """Produces a random vector of inputs."""
        return np.random.random( self.get_param_count() )
        # return [ np.pi ] * self.get_param_count()

    @abc.abstractmethod
    def get_location ( self, x ):
        """Returns the gate's location."""
        pass

    @abc.abstractmethod
    def get_param_count ( self ):
        """Returns the number of the gate's input parameters."""
        pass

    @abc.abstractmethod
    def get_matrix ( self, x ):
        """Produces the circuit matrix for this gate."""
        pass

    @abc.abstractmethod
    def get_gate_matrix ( self, x ):
        """Produces the matrix for this gate on its own."""
        pass

    @abc.abstractmethod
    def get_matrix_and_derivatives ( self, x ):
        """Produces the circuit matrix and partials for this gate."""
        pass

