"""
This module implements the GenericGate Class.

A GenericGate represents a gate with arbitrary location and function.
"""

import numpy as np
import scipy.linalg as la
import tensorflow as tf
import itertools as it

from pauli import *


class GenericGate():
    """
    The GenericGate Class.
    """

    def __init__ ( self, name, num_qubits, gate_size,
                   init_values = None, parity = None ):
        """
        GenericGate Class Constructor

        Args:
            name (str): The name of the layer

            num_qubits (int): The number of qubits in the circuit

            gate_size (int): The size of the gates

            init_values (List[float]): Initial values for the layer

            parity (int): The side of the topology to occupy. Can be
                          either 0 or 1; prevents consecutive layers
                          from choosing the same link
        """

        if gate_size > num_qubits:
            raise ValueError( "Gate Size must be <= to number of qubits." )

        self.name        = name
        self.num_qubits  = num_qubits
        self.gate_size   = gate_size
        self.init_values = init_values
        self.topology    = list( it.combinations( range( self.num_qubits ),
                                                  self.gate_size ) )

        if parity == 0:
            self.topology = self.topology[:len(self.topology)//2]
        elif parity == 1:
            self.topology = self.topology[len(self.topology)//2:]

        self.num_link_vars = len( self.topology )
        self.num_gate_vars = 4 ** self.gate_size

        if self.init_values is None:
            self.init_values = ( [0] * self.num_link_vars ) + \
                               ( [np.sqrt( self.num_gate_vars ** -1 )] *
                                 self.num_gate_vars )

        if len( self.init_values ) != self.num_link_vars + self.num_gate_vars:
            raise ValueError( "Incorrect length of init_values." )

        # Construct layer
        with tf.variable_scope( self.name ):

            self.variables = [ tf.Variable( val, dtype = tf.float64 )
                               for val in self.init_values ]

            self.link_vars = self.variables[:self.num_link_vars]
            self.gate_vars = self.variables[self.num_link_vars:]
            self.cast_vars = [ tf.cast( x, tf.complex128 ) for x in self.gate_vars ]

            self.tensors = []

            for link in self.topology:
                paulis = get_pauli_n_qubit_projection( self.num_qubits, link )
                H = tf.reduce_sum( [ var * pauli for var, pauli
                                     in zip( self.cast_vars, paulis ) ], 0 )
                self.tensors.append( H )

            link_exps = [ tf.exp( 500 * var )
                          for var in self.link_vars ]
            sum_exp = tf.reduce_sum( link_exps ) + 1e-8
            self.softmax = [ link_exp / sum_exp for link_exp in link_exps ]
            self.cast_max = [ tf.cast( softmax_var, tf.complex128 )
                              for softmax_var in self.softmax ]
            self.layer = tf.reduce_sum( [
                                        softvar * gate
                                        for softvar, gate
                                        in zip( self.cast_max, self.tensors )
                                        ], 0 )
            self.unitary = tf.linalg.expm( 1j  * self.layer )

    def get_tensor ( self ):
        return self.layer

    def get_unitary_tensor ( self ):
        return self.unitary

    def get_variables ( self ):
        return self.variables

    def get_gate_vars ( self ):
        return self.gate_vars

    def get_softmax_vars ( self ):
        return self.softmax

    def get_link_vars ( self ):
        return self.link_vars

    def get_values ( self, sess ):
        return sess.run( self.variables )

    def get_link_vals ( self, sess ):
        return sess.run( self.link_vars )

    def get_link ( self, sess ):
        link_idx = np.argmax( sess.run( self.link_vars ) )
        return self.topology[ link_idx ]

    def get_gate_vals ( self, sess ):
        return sess.run( self.gate_vars )

    def get_unitary ( self, sess ):
        link = self.get_link( sess )
        unitary_params = self.get_gate_vals( sess )
        paulis = get_pauli_n_qubit_projection( self.num_qubits, link )
        H = np.sum( [ u*p for u, p in zip( unitary_params, paulis ) ], 0 )
        return la.expm( 1j * H )
