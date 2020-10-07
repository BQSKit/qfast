"""
This module defines the CircuitModel abstract base class.

A CircuitModel models a large unitary as a circuit of gates.
This enables the decomposition of unitaries.

All model plugins must extend this class and implement the
functionality outlined here.
"""

import abc

import numpy as np
import functools as ft

import qfast
from qfast import utils
from qfast.gate import Gate
from qfast.decomposition.gatemodel import GateModel


class ModelMeta ( abc.ABCMeta ):
    """The CircuitModel Metaclass."""

    def __init__ ( cls, name, bases, attr ):
        """Automatically registers model plugins with qfast."""

        qfast.modelsubclasses[name] = cls
        super().__init__( name, bases, attr )


class CircuitModel ( metaclass = ModelMeta ):
    """The CircuitModel abstract base class."""

    def __init__ ( self, utry, gate_size, locations, optimizer,
                   success_threshold = 1e-3, partial_solution_callback = None ):
        """
        Default constructor for CircuitModels.

        Args:
            utry (np.ndarray): The unitary to model.

            gate_size (int): The size of the model's gates.

            locations (List[Tuple[int]]): The valid locations for gates.

            optimizer (Optimizer): The optimizer available for use.

            success_threshold (float): The distance criteria for success.

            partial_solution_callback (None or callable): callback for
                partial solutions. If not None, then callable that takes
                a list[gate.Gate] and returns nothing.

        Raises:
            ValueError: If the gate_size or locations are invalid.
        """

        if partial_solution_callback is not None:
            if not callable( partial_solution_callback ):
                raise TypeError( "Invalid partial_solution_callback." )

        self.num_qubits = utils.get_num_qubits( utry )

        if gate_size >= self.num_qubits:
            raise ValueError( "Invalid gate_size" )

        self.gate_size = gate_size

        if not utils.is_valid_locations( locations, self.num_qubits,
                                         self.gate_size ):
            raise TypeError( "Invalid locations" )

        self.partial_solution_callback = partial_solution_callback
        self.locations = locations
        self.optimizer = optimizer
        self.utry = utry
        self.utry_dag = utry.conj().T
        self.success_threshold = success_threshold
        self.gates = []
        self.param_ranges = [ 0 ]
        self.x = self.get_initial_input()

    @abc.abstractmethod
    def solve ( self ):
        """
        Solve the decomposition problem defined in this CircuitModel.

        Returns:
            (List[Gate]): The list of gates that implement the unitary.
        """

    def get_initial_input ( self ):
        """Get initial input of model."""
        if self.depth() == 0:
            return np.array([])

        return np.concatenate( [ gate.get_initial_input()
                                 for gate in self.gates ] )

    def reset_input ( self ):
        """Resets input and recalculates parameter ranges."""
        self.param_ranges = [ 0 ]

        for gate in self.gates:
            self.param_ranges.append( self.param_ranges[-1]
                                      + gate.get_param_count() )

        self.x = self.get_initial_input()

    def append_gate ( self, gate, init_input = None ):
        """Append a gate onto the model."""

        if not isinstance( gate, GateModel ):
            raise TypeError( "Gate is not a model gate.""" )

        self.gates.append( gate )
        self.param_ranges.append( self.param_ranges[-1]
                                  + gate.get_param_count() )
        if init_input is not None:
            self.x = np.concatenate( ( self.x, init_input ) )
        else:
            self.x = np.concatenate( ( self.x, gate.get_initial_input() ) )

    def insert_gate ( self, idx, gate, init_input = None ):
        """Insert a gate into the model."""

        if not isinstance( gate, GateModel ):
            raise TypeError( "Gate is not a model gate.""" )

        self.gates.insert( idx, gate )

        idx = len( self.gates ) + idx if idx < 0 else idx
        if init_input is not None:
            self.x = np.concatenate( ( self.x[ : self.param_ranges[ idx ] ],
                                       init_input,
                                       self.x[ self.param_ranges[ idx ] : ] ) )
        else:
            self.x = np.concatenate( ( self.x[ : self.param_ranges[ idx ] ],
                                       gate.get_initial_input(),
                                       self.x[ self.param_ranges[ idx ] : ] ) )

        self.param_ranges = [ 0 ]
        for gate in self.gates:
            self.param_ranges.append( self.param_ranges[-1]
                                      + gate.get_param_count() )

    def pop_gate ( self ):
        """Remove and return the last gate in model."""

        if len( self.gates ) <= 0:
            raise IndexError( "No gates in model to pop." )

        self.x = self.x[ : self.param_ranges[-2] ]
        self.param_ranges.pop()
        return self.gates.pop()

    def get_param_count ( self ):
        """Total number of parameters in model."""
        return self.param_ranges[-1]

    def depth ( self ):
        """Returns depth of model."""
        return len( self.gates )

    def distance ( self ):
        """Calculates the model's distance to the target unitary."""
        M = self.get_matrix( self.x )
        num = np.abs( np.trace( self.utry_dag @ M ) )
        dem = M.shape[0]
        return 1 - ( num / dem )

    def success ( self ):
        """If the model has successfully modeled the target unitary."""
        if self.partial_solution_callback is not None:
            self.partial_solution_callback( self.get_gate_list() )

        return self.distance() < self.success_threshold

    def get_input_slice ( self, gate_idx ):
        """Returns the inputs for a specific gate."""
        if gate_idx < 0:
            lower_bound = self.param_ranges[ gate_idx - 1 ]
            upper_bound = self.param_ranges[ gate_idx ]
        else:
            lower_bound = self.param_ranges[ gate_idx ]
            upper_bound = self.param_ranges[ gate_idx + 1 ]

        return self.x[ lower_bound : upper_bound ]

    def get_gate_list ( self ):
        """Converts model to list of qfast gate objects."""
        gate_list = []

        for i, gate in enumerate( self.gates ):
            M = gate.get_gate_matrix( self.get_input_slice( i ) )
            L = gate.get_location( self.get_input_slice( i ) )
            gate_list.append( Gate( M, L ) )

        return gate_list

    def get_matrix ( self, x ):
        """Returns the circuit model's matrix."""
        if len( self.gates ) == 0:
            return np.identity( self.utry_dag.shape[0] )

        if len( self.gates ) == 1:
            return self.gates[0].get_matrix(x)

        matrices = []

        for i, gate in enumerate( self.gates ):
            matrices.append( gate.get_matrix( self.get_input_slice( i ) ) )

        return ft.reduce( np.matmul, reversed( matrices ) )

    def get_matrix_and_derivatives ( self, x ):
        """Returns the circuit model's matrix and derivatives."""
        if len( self.gates ) == 0:
            return np.identity( self.utry_dag.shape[0] ), np.array([])

        if len( self.gates ) == 1:
            return self.gates[0].get_matrix_and_derivatives(x)

        matrices = []
        derivatives = []

        for i, gate in enumerate( self.gates ):
            lower_bound = self.param_ranges[ i ]
            upper_bound = self.param_ranges[ i + 1 ]
            x_slice = x[ lower_bound : upper_bound ]
            M, J = gate.get_matrix_and_derivatives( x_slice )
            matrices.append( M )
            derivatives.append( J )

        matrix = ft.reduce( np.matmul, reversed( matrices ) )
        jacs = []

        for i, dM in enumerate( derivatives ):

            if i + 1 < len( derivatives ):
                left = ft.reduce( np.matmul, reversed( matrices[i+1:] ) )
            else:
                left = np.identity( self.utry_dag.shape[0] )


            if i != 0:
                right = ft.reduce( np.matmul, reversed( matrices[:i] ) )
            else:
                right = np.identity( self.utry_dag.shape[0] )

            for dm in dM:
                jacs.append( left @ dm @ right )

        return matrix, np.array( jacs )

    def objective_fn ( self, x ):
        """The objective function of the optimizer."""
        M, dM = self.get_matrix_and_derivatives( x )
        obj = -np.real( np.trace( self.utry_dag @ M ) )
        jacs = []
        for dm in dM:
            jacs.append( -np.real( np.trace( self.utry_dag @ dm ) ) )
        jacs = np.array( jacs )
        return obj, jacs

    def optimize ( self, fine = False ):
        """Perform an optimizer call."""
        if fine:
            self.x = self.optimizer.minimize_fine( self.objective_fn, self.x )
        else:
            self.x = self.optimizer.minimize_coarse( self.objective_fn, self.x )

