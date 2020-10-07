"""
QFAST Soft Pauli Model Module

This models a circuit as a sequence of fixed gates potentially led by
a generic gate. Generic gates multiplex gate placement, or location.
"""


import logging

import numpy     as np
import functools as ft

from qfast import utils
from qfast import gate
from qfast.decomposition.circuitmodel import CircuitModel
from qfast.decomposition.models.softpauli.genericgate import GenericGate
from qfast.decomposition.models.softpauli.fixedgate import FixedGate


logger = logging.getLogger( "qfast" )


class SoftPauliModel ( CircuitModel ):

    def __init__ ( self, utry, gate_size, locations, optimizer,
                   success_threshold = 1e-3, partial_solution_callback = None,
                   progress_threshold = 5e-3 ):
        """
        Soft Pauli Model Constructor

        Args:
            utry (np.ndarray): The unitary to model.

            gate_size (int): The size of the model's gate.

            locations (list[tuple[int]): The valid locations for gates.

            optimizer (Optimizer): The optimizer available for use.

            success_threshold (float): The distance criteria for success.

            partial_solution_callback (None or callable): callback for
                partial solutions. If not None, then callable that takes
                a list[gate.Gate] and returns nothing.

            progress_threshold (float): The distance increase criteria
                for successful expansion.
        """

        super().__init__( utry, gate_size, locations, optimizer,
                          success_threshold, partial_solution_callback )

        self.progress_threshold = progress_threshold

        self.head = GenericGate( self.num_qubits, self.gate_size,
                                 self.locations )
        self.append_gate( self.head )
        self.last_dist = 1

    def progress ( self ):
        """If the model has made progress."""
        return self.last_dist - self.distance() > self.progress_threshold

    def expand ( self, location ):
        """Expand the model by adding gates."""
        logger.info( "Expanding by adding a gate at location %s"
                     % str( location ) )

        new_gate = FixedGate( self.num_qubits, self.gate_size, location )
        self.insert_gate( -1, new_gate )
        self.head.lift_restrictions()
        self.head.restrict( location )

    def finalize ( self ):
        """Finalize the circuit by replacing the head if necessary."""
        location = self.head.get_location( self.get_input_slice( -1 ) )
        fun_vals = self.head.get_function_values( self.get_input_slice( -1 ),
                                                  True )
        self.pop_gate()

        new_gate = FixedGate( self.num_qubits, self.gate_size, location )
        self.append_gate( new_gate, fun_vals )
        self.optimize( fine = True )

        return self.get_gate_list()

    def solve ( self ):
        """Solve the model for the target unitary."""
        failed_locs = []

        while True:

            self.reset_input()

            self.optimize()

            logger.info( "Finished optimizing depth %d at %e distance."
                         % ( self.depth(), self.distance() ) )

            if self.success():
                logger.info( "Exploration finished: success" )
                return self.finalize()

            location = self.head.get_location( self.get_input_slice( -1 ) )

            if self.progress():
                logger.info( "Progress has been made, depth increasing." )
                self.last_dist = self.distance()
                self.expand( location )

            elif self.head.cannot_restrict():
                logger.info( "Progress has not been made." )
                logger.info( "Cannot restrict further, depth increasing." )

                failed_locs.append( ( location, self.distance() ) )

                if len( failed_locs ) > 0:
                    failed_locs.sort( key = lambda x : x[1] )
                    location, self.last_dist = failed_locs[0]
                else:
                    self.last_dist = self.distance()

                self.expand( location )
                failed_locs = []

            else:
                logger.info( "Progress has not been made, restricting model." )
                failed_locs.append( ( location, self.distance() ) )
                self.head.restrict( location )

        return self.finalize()

