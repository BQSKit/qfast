"""
This module implements the GenericGate Class.

A GenericGate represents a gate with variable location and function.
"""

import tensorflow   as tf
import numpy        as np
import scipy.linalg as la
import itertools    as it

from .pauli import get_pauli_n_qubit_projection
from .pauli import get_pauli_tensor_n_qubit_projection


class GenericGate():
    """
    The GenericGate Class.
    """

    def __init__ ( self, name, num_qubits, gate_size, lm, fun_vals = None,
                   loc_vals = None, parity = None ):
        """
        GenericGate Class Constructor.

        Args:
            name (str): The name of the gate

            num_qubits (int): The number of qubits in the circuit

            gate_size (int): The size of the gate

            lm (LocationModel): The model that maps loc_vals to locations

            fun_vals (List[float]): Initial values for the
                                    gate's function

            loc_vals (List[float]): Initial values for the
                                    gate's location

            parity (int): The side of the topology to occupy. Can be
                          either 0 or 1; prevents consecutive gates
                          from choosing the same location
        """

        if gate_size > num_qubits:
            raise ValueError( "Gate Size must be <= to number of qubits." )

        self.name       = name
        self.num_qubits = num_qubits
        self.gate_size  = gate_size
        self.loc_vals   = loc_vals
        self.fun_vals   = fun_vals
        self.topology   = list( lm.locations ) if parity is None else lm.buckets[ parity ]

        self.num_loc_vars = len( self.topology )
        self.num_fun_vars = 4 ** self.gate_size

        if self.fun_vals is None:
            self.fun_vals = [ np.sqrt( self.num_fun_vars ** -1 ) ] * \
                              self.num_fun_vars

        if self.loc_vals is None:
            self.loc_vals = [0] * self.num_loc_vars

        if len( self.fun_vals ) != self.num_fun_vars:
            raise ValueError( "Incorrect number of function values." )

        if len( self.loc_vals ) != self.num_loc_vars:
            raise ValueError( "Incorrect number of location values." )

        # Construct Tensor
        with tf.variable_scope( self.name ):

            self.fun_vars = [ tf.Variable( val, dtype = tf.float64 )
                              for val in self.fun_vals ]

            self.loc_vars = [ tf.Variable( val, dtype = tf.float64 )
                              for val in self.loc_vals ]

            self.cast_vars = [ tf.cast( x, tf.complex128 )
                               for x in self.fun_vars ]

            gates = []

            for location in self.topology:
                paulis = get_pauli_tensor_n_qubit_projection( self.num_qubits,
                                                              location )

                H = tf.reduce_sum( [ var * pauli for var, pauli
                                     in zip( self.cast_vars, paulis ) ], 0 )

                gates.append( H )

            loc_exps = [ tf.exp( 500 * var ) for var in self.loc_vars ]

            sum_exp = tf.reduce_sum( loc_exps ) + 1e-15

            self.softmax = [ loc_exp / sum_exp for loc_exp in loc_exps ]

            self.cast_max = [ tf.cast( softmax_var, tf.complex128 )
                              for softmax_var in self.softmax ]

            self.herm = tf.reduce_sum( [ softvar * gate
                                         for softvar, gate
                                         in zip( self.cast_max, gates )
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

    def get_fun_vals ( self, sess ):
        return sess.run( self.fun_vars )

    def get_loc_vals ( self, sess ):
        return sess.run( self.loc_vars )

    def get_location ( self, sess ):
        loc_idx = np.argmax( sess.run( self.loc_vars ) )
        return self.topology[ loc_idx ]

    def get_unitary ( self, sess ):
        location = self.get_location( sess )
        fun_params = self.get_fun_vals( sess )
        paulis = get_pauli_n_qubit_projection( self.num_qubits, location )
        H = np.sum( [ a*p for a, p in zip( fun_params, paulis ) ], 0 )
        return la.expm( 1j * H )
