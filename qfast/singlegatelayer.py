"""
This module implements the MultiGateLayer Class.

A MultiGateLayer represents one arbitrary n-qubit gate on one qubit tuple.
"""

import numpy as np
import scipy.linalg as la
import tensorflow as tf

from .pauli import *


class SingleGateLayer():
    """
    The SingleGateLayer Layer Class.
    """

    def __init__ ( self, name, num_qubits, gate_size, link, init_values = None ):
        """
        SingleGateLayer Class Constructor

        Args:
            name (str): The name of the layer

            num_qubits (int): The number of qubits in the circuit

            gate_size (int): The size of the gates

            link (Tuple[int]) The qubits this gate acts on

            init_values (List[float]): Initial values for the layer
        """

        if gate_size > num_qubits:
            raise ValueError( "Gate Size must be <= to number of qubits." )

        self.name        = name
        self.num_qubits  = num_qubits
        self.gate_size   = gate_size
        self.link        = link
        self.init_values = init_values

        self.num_gate_vars = 4 ** self.gate_size

        if self.init_values is None:
            self.init_values = [np.sqrt( self.num_gate_vars ** -1 )] * self.num_gate_vars

        if len( self.init_values ) != self.num_gate_vars:
            raise ValueError( "Incorrect length of init_values." )

        # Construct layer
        with tf.variable_scope( self.name ):

            self.variables = [ tf.Variable( val, dtype = tf.float64 )
                               for val in self.init_values ]

            self.cast_vars = [ tf.cast( x, tf.complex128 ) for x in self.variables ]

            self.tensors = []

            paulis = get_pauli_n_qubit_projection( self.num_qubits, self.link )
            assert( len( paulis ) == (4 ** self.gate_size) )
            self.layer = tf.reduce_sum( [ var * pauli for var, pauli, in zip( self.cast_vars, paulis ) ], 0 )
            self.unitary = tf.linalg.expm( 1j * self.layer )

    def get_tensor ( self ):
        return self.layer

    def get_unitary_tensor ( self ):
        return self.unitary

    def get_variables ( self ):
        return self.variables

    def get_gate_vars ( self ):
        return self.variables

    def get_values ( self, sess ):
        return sess.run( self.variables )

    def get_link ( self, sess ):
        return self.link

    def get_gate_vals ( self, sess ):
        return sess.run( self.variables )

    def get_unitary ( self, sess ):
        link = self.get_link( sess )
        unitary_params = self.get_gate_vals( sess )
        paulis = get_pauli_n_qubit_projection( self.num_qubits, link )
        H = np.sum( [ u*p for u, p in zip( unitary_params, paulis ) ], 0 )
        return la.expm( 1j * H )
