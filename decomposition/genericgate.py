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

    def __init__ ( self, name, num_qubits, gate_size, fun_values = None,
                   loc_values = None, parity = None ):
        """
        GenericGate Class Constructor.

        Args:
            name (str): The name of the gate

            num_qubits (int): The number of qubits in the circuit

            gate_size (int): The size of the gate

            fun_values (List[float]): Initial values for the
                                      gate's function

            loc_values (List[float]): Initial values for the
                                      gate's location

            parity (int): The side of the topology to occupy. Can be
                          either 0 or 1; prevents consecutive gates
                          from choosing the same loc
        """

        if gate_size > num_qubits:
            raise ValueError( "Gate Size must be <= to number of qubits." )

        self.name       = name
        self.num_qubits = num_qubits
        self.gate_size  = gate_size
        self.loc_values = loc_values
        self.fun_values = fun_values
        self.topology   = list( it.combinations( range( self.num_qubits ),
                                                 self.gate_size ) )

        if parity == 0:
            self.topology = self.topology[:len(self.topology)//2]
        elif parity == 1:
            self.topology = self.topology[len(self.topology)//2:]

        self.num_loc_vars = len( self.topology )
        self.num_fun_vars = 4 ** self.gate_size

        if self.fun_values is None:
            self.fun_values = [np.sqrt( self.num_fun_vars ** -1 )] *
                               self.num_fun_vars

        if self.loc_values is None:
            self.loc_values = [0] * self.num_loc_vars

        if len( self.fun_values ) != self.num_fun_vars:
            raise ValueError( "Incorrect number of function values." )

        if len( self.loc_values ) != self.num_loc_vars:
            raise ValueError( "Incorrect number of location values." )

        # Construct Tensor
        with tf.variable_scope( self.name ):

            self.fun_vars = [ tf.Variable( val, dtype = tf.float64 )
                              for val in self.fun_values ]

            self.loc_vars = [ tf.Variable( val, dtype = tf.float64 )
                              for val in self.loc_values ]

            self.cast_vars = [ tf.cast( x, tf.complex128 )
                               for x in self.fun_vars ]

            self.tensors = []

            for link in self.topology:
                paulis = get_pauli_n_qubit_projection( self.num_qubits, link )
                H = tf.reduce_sum( [ var * pauli for var, pauli
                                     in zip( self.cast_vars, paulis ) ], 0 )
                self.tensors.append( H )

            loc_exps = [ tf.exp( 500 * var )
                         for var in self.loc_vars ]

            sum_exp = tf.reduce_sum( loc_exps ) + 1e-8

            self.softmax = [ loc_exp / sum_exp for loc_exp in loc_exps ]

            self.cast_max = [ tf.cast( softmax_var, tf.complex128 )
                              for softmax_var in self.softmax ]

            self.herm = tf.reduce_sum( [ softvar * gate
                                         for softvar, gate
                                         in zip( self.cast_max, self.tensors )
                                       ], 0 )

            self.gate = tf.linalg.expm( 1j  * self.herm )

    def get_herm ( self ):
        return self.herm

    def get_gate ( self ):
        return self.gate

    def get_tensor ( self ):
        return self.gate

    def get_fun_vars ( self ):
        return self.fun_vars

    def get_loc_vars ( self ):
        return self.loc_vars

    def get_softmax_vars ( self ):
        return self.softmax

    def get_fun_vals ( self, sess ):
        return sess.run( self.fun_vars )

    def get_loc_vals ( self, sess ):
        return sess.run( self.loc_vars )

    def get_location ( self, sess ):
        loc_idx = np.argmax( sess.run( self.loc_vars ) )
        return self.topology[ loc_idx ]

    # def get_unitary ( self, sess ):
    #     loc = self.get_location( sess )
    #     unitary_params = self.get_gate_vals( sess )
    #     paulis = get_pauli_n_qubit_projection( self.num_qubits, loc )
    #     H = np.sum( [ u*p for u, p in zip( unitary_params, paulis ) ], 0 )
    #     return la.expm( 1j * H )
