"""
QFAST Permutation Fixed Structure Model Module

This models a circuit as a sequence of fixed gates in a fixed pattern.
"""


import logging

import numpy     as np
import functools as ft

from qfast import utils
from qfast.decomposition.circuitmodel import CircuitModel
from qfast.decomposition.models.perm.fixedgate import FixedGate


logger = logging.getLogger( "qfast" )


class FixedModel ( CircuitModel ):

    def __init__ ( self, utry, gate_size, locations, optimizer,
                   success_threshold = 1e-3, partial_solution_callback = None,
                   structure = None, repeat = False ):
        """
        Fixed Structure Model Constructor

        Args:
            utry (np.ndarray): The unitary to model.

            gate_size (int): The size of the model's gate.

            locations (list[tuple[int]): The valid locations for gates.

            optimizer (Optimizer): The optimizer available for use.

            success_threshold (float): The distance criteria for success.

            partial_solution_callback (None or callable): callback for
                partial solutions. If not None, then callable that takes
                a list[gate.Gate] and returns nothing.

            structure (list[tuple[int]]): The initial structure of the
                model.

            repeat (bool): If true, repeat structure until success.
        """

        super().__init__( utry, gate_size, locations, optimizer,
                          success_threshold, partial_solution_callback )

        if structure is None:
            raise ValueError( "Must include structure." )

        if not utils.is_valid_locations( structure, self.num_qubits,
                                         self.gate_size ):
            raise TypeError( "Invalid locations." )

        self.structure = structure

        for location in self.structure:
            gate = FixedGate( self.num_qubits, self.gate_size, location )
            self.append_gate( gate )

        self.repeat = repeat

    def solve ( self ):

        while True:

            self.optimize( fine = True )

            if self.success():
                logger.info( "Successfully completed at distance: %e"
                             % self.distance() )
                return self.get_gate_list()

            if not self.repeat:
                logger.info( "Unsuccessfully completed at distance: %e"
                             % self.distance() )
                return self.get_gate_list()

            logger.info( "Finished depth %d at distance: %e"
                         % ( self.depth(), self.distance() ) )
        
            for location in self.structure:
                gate = FixedGate( self.num_qubits, self.gate_size, location )
                self.append_gate( gate )

