"""
This module implements the FixedGate Class.

A FixedGate represents a gate with fixed location and variable function.
"""

import tensorflow   as tf
import numpy        as np
import scipy.linalg as la

from .pauli import get_pauli_n_qubit_projection
from .pauli import get_pauli_tensor_n_qubit_projection


class FixedGate():
    """
    The FixedGate Class.
    """

    def __init__ ( self, name, num_qubits, gate_size, loc, fun_vals = None ):
        """
        FixedGate Class Constructor.

        Args:
            name (str): The name of the layer

            num_qubits (int): The number of qubits in the circuit

            gate_size (int): The size of the gate

            loc (Tuple[int]) The qubits this gate acts on (location)

            fun_vals (List[float]): Initial values for the gate's
                                    function
        """

        if gate_size > num_qubits:
            raise ValueError( "Gate Size must be <= to number of qubits." )

        self.name       = name
        self.num_qubits = num_qubits
        self.gate_size  = gate_size
        self.location   = loc
        self.fun_vals   = fun_vals

        self.num_fun_vars = 4 ** self.gate_size

        if self.fun_vals is None:
            self.fun_vals = [ np.sqrt( self.num_fun_vars ** -1 ) ] * \
                              self.num_fun_vars

        if len( self.fun_vals ) != self.num_fun_vars:
            raise ValueError( "Incorrect number of function values." )

        # Construct Tensor
        with tf.variable_scope( self.name ):

            self.fun_vars = [ tf.Variable( val, dtype = tf.float64 )
                              for val in self.fun_vals ]

            self.cast_vars = [ tf.cast( x, tf.complex128 )
                               for x in self.fun_vars ]

            paulis = get_pauli_tensor_n_qubit_projection( self.num_qubits,
                                                          self.location )

            self.herm = tf.reduce_sum( [ var * pauli
                                         for var, pauli
                                         in zip( self.cast_vars, paulis )
                                       ], 0 )

            self.gate = tf.linalg.expm( 1j * self.herm )

    def get_herm ( self ):
        return self.herm

    def get_gate ( self ):
        return self.gate

    def get_tensor ( self ):
        return self.gate

    def get_fun_vars ( self ):
        return self.fun_vars

    def get_fun_vals ( self, sess ):
        return sess.run( self.fun_vars )

    def get_location ( self ):
        return self.location

    def get_unitary ( self, sess ):
        fun_params = self.get_fun_vals( sess )
        paulis = get_pauli_n_qubit_projection( self.num_qubits, self.location )
        H = np.sum( [ a*p for a, p in zip( fun_params, paulis ) ], 0 )
        return la.expm( 1j * H )
