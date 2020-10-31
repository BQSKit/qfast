"""
This module implements the Decomposer class.

The decomposer uses a circuit model to break a large unitary into 
smaller ones.
"""


import numpy as np
import itertools as it

from qfast import plugins
from qfast import utils
from qfast.gate import Gate
from qfast.topology import Topology

import logging
logger = logging.getLogger( "qfast" )


class Decomposer():

    def __init__ ( self, utry, target_gate_size = 2, model = "PermModel",
                   optimizer = "LBFGSOptimizer",
                   hierarchy_fn = lambda x : x // 3 if x > 5 else 2,
                   coupling_graph = None, intermediate_solution_callback = None,
                   model_options = {} ):
        """
        Initializes a decomposer.

        Args:
            utry (np.ndarray): A unitary matrix to decompose

            target_gate_size (int): After decomposition, this will be
                the largest size of any gate in the returned list.

            model (str): The circuit model to use during decomposition.

            optimizer (str): The optimizer to use during decomposition.

            hierarchy_fn (callable): This function determines the
                decomposition hierarchy.

            coupling_graph (None or list[tuple[int]]): Determines the
                connection of qubits. If none, will be set to all-to-all.

            intermediate_solution_callback (None or callable): Callback
                function for intermediate solutions. If not None, then
                a function that takes in a list[Gates] and returns nothing.

        Raises:
            ValueError: If the target_gate_size is nonpositive or too large.

            RuntimeError: If the model or optimizer cannot be found.
        """

        if not utils.is_unitary( utry, tol = 1e-14 ):
            logger.warning( "Unitary is not doubly-precise." )
            logger.warning( "Proceeding with closest unitary to input." )
            self.utry = utils.closest_unitary( utry )
        else:
            self.utry = utry

        self.num_qubits = utils.get_num_qubits( utry )

        if target_gate_size <= 0 or target_gate_size > self.num_qubits:
            raise ValueError( "Invalid target gate size." )

        self.target_gate_size = target_gate_size

        if not callable( hierarchy_fn ):
            raise TypeError( "Invalid hierarchy function." )

        if intermediate_solution_callback is not None:
            if not callable( intermediate_solution_callback ):
                raise TypeError( "Invalid intermediate solution callback." )

        self.hierarchy_fn = hierarchy_fn
        self.intermediate_solution_callback = intermediate_solution_callback
        self.topology = Topology( self.num_qubits, coupling_graph )

        if model not in plugins.get_models():
            raise RuntimeError( f"Cannot find decomposition model: {model}" )

        self.model = plugins.get_model( model )

        if optimizer not in plugins.get_optimizers():
            raise RuntimeError( f"Cannot find optimizer: {optimizer}" )

        self.model_options = model_options
        self.optimizer = plugins.get_optimizer( optimizer )

        logger.debug( "Created decomposer with %s and %s."
                      % ( model, optimizer ) )

    def decompose ( self ):
        """
        Performs the decomposition phase.

        Returns:
            (list[gate.Gate]): List of gates that implements the
                decomposer's unitary. Each gate will have a size less than
                or equal to the target gate size.
        """

        gate_list = [ Gate( self.utry, tuple( range( self.num_qubits ) ) ) ] 

        while any( [ gate.num_qubits > self.target_gate_size
                     for gate in gate_list ] ):

            new_gate_list = []

            for gate in gate_list:

                if gate.num_qubits <= self.target_gate_size:
                    new_gate_list.append( gate )
                else:
                    next_gate_size = self.hierarchy_fn( gate.num_qubits )
                    t = self.topology.get_locations( next_gate_size )
                    m = self.model( self.utry, next_gate_size, t, self.optimizer(), **self.model_options )
                    new_gate_list += m.solve()

            gate_list = new_gate_list

            if self.intermediate_solution_callback is not None:
                self.intermediate_solution_callback( gate_list )

        return gate_list

