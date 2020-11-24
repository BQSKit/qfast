"""
QFAST Prediction Model Module

This model builds a circuit by predicting next gate location.
"""


import logging

import copy
import numpy     as np
import functools as ft
import itertools as it

from scipy.stats import unitary_group

from qfast import utils
from qfast.pauli import *
from qfast.gate import Gate
from qfast.decomposition.circuitmodel import CircuitModel
from qfast.decomposition.models.perm.fixedgate import FixedGate

import qfactor


logger = logging.getLogger( "qfast" )


class PredictModel ( CircuitModel ):

    def __init__ ( self, utry, gate_size, topology, optimizer,
                   success_threshold = 1e-6, partial_solution_callback = None,
                   larger_gate_size = None, reduction_level = 1,
                   min_iters = 0, diff_tol_r = 1e-4 ):
        """
        Fixed Structure Model Constructor

        Args:
            utry (np.ndarray): The unitary to model.

            gate_size (int): The size of the model's gate.

            topology (Topology): The circuit topology.

            optimizer (Optimizer): The optimizer available for use.

            success_threshold (float): The distance criteria for success.

            partial_solution_callback (None or callable): callback for
                partial solutions. If not None, then callable that takes
                a list[gate.Gate] and returns nothing.

            larger_gate_size (None or int): The gate size used when
                the regular gate size doesn't improve the score.

            reduction_level (int): If 1 or 2, will run a post processing
                step that attempts to reduce the depth of the circuit.
        """

        super().__init__( utry, gate_size, topology, optimizer,
                          success_threshold, partial_solution_callback )

        self.larger_gate_size = larger_gate_size or self.gate_size + 1

        if self.larger_gate_size >= self.num_qubits:
            raise ValueError( "Too large of a larger gate size." )

        self.larger_locs = self.topology.get_locations( self.larger_gate_size )
        self.reduction_level = reduction_level

        # Optimizer Settings
        self.min_iters = min_iters
        self.diff_tol_r = diff_tol_r

    def sort_locations ( self, alpha, locations ):
        """
        Sorts the locations in t based on projected magnitude in alpha.

        Args:
            alpha (np.ndarray): Pauli Expansion Coefficients.

            locations (list[tuple[int]]): A set of locations to sort.

        Returns:
            (list[tuple[int]]): Sorted locations.
        """

        location_sums = []
        for location in locations:
            sum = 0
            for ps in it.product( [ 1, 2, 3 ], repeat = len( location ) ):
                idx = 0
                for p, q in zip( ps, location ):
                    idx += p * ( 4 ** ( self.num_qubits - q - 1 ) )
                sum += np.abs( alpha[ idx ] )
            location_sums.append( ( location, sum ) )

        sorted_locations = sorted( location_sums, key = lambda x : x[1],
                                   reverse = True )

        sorted_locations, _ = zip( *sorted_locations )
        return sorted_locations

    def solve ( self ):

        circuit = []
        last_ls = []
        last_d = None

        while True:
            # Chose location
            ct = qfactor.tensors.CircuitTensor( self.utry, circuit )
            paulis = pauli_expansion( unitary_log_no_i( ct.utry, tol = 1e-12 ) )
            if len( last_ls ) >= 2:
                ls = self.sort_locations( paulis, self.larger_locs )
            else:
                ls = self.sort_locations( paulis, self.locations )

            for i in range( len( ls ) ):
                l = ls[i]
                if l not in last_ls:
                  break 

            # Expand Circuit
            circuit.append( qfactor.Gate( unitary_group.rvs( 2 ** len( l ) ), l ) )

            # Optimize
            qfactor.optimize( circuit, self.utry, min_iters = self.min_iters,
                              diff_tol_r = self.diff_tol_r,
                              dist_tol = self.success_threshold )

            dist = qfactor.get_distance( circuit, self.utry )
            logger.info( "Finished adding gate at location %s with distance %f"
                         % ( str( l ), dist ) )

            # Check for success
            if dist < self.success_threshold:
                if self.reduction_level == 1:
                    return self.greedy_dim_reduce( circuit )
                elif self.reduction_level >= 2:
                    return self.dim_reduce( circuit )
                else:
                    return [ Gate( g.utry, g.location ) for g in circuit ]

            # Check for stagnation
            if last_d and np.abs( dist - last_d ) < 1e-6 + 1e-3 * dist: # self.progress_threshold abs + rel
                logger.info( "Insufficient improvement; popping gate." )
                circuit.pop( -1 )
                last_ls.append( l )
                continue

            last_d = dist
            last_ls = [ l ]

            if self.partial_solution_callback:
                self.partial_solution_callback( circuit )

    def greedy_dim_reduce ( self, circuit ):
        logger.info( "Starting Dimension Reduction" )

        # Initialize Population
        pop = [ ( [], circuit ) ]

        while len( pop ) > 0:
            # Asexual Breeding
            logger.info( "Breeding..." )
            next_gen = []
            for circ in pop:
                for i in range( len( circ[1] ) ):
                    new_circ = copy.deepcopy( circ )
                    new_circ[1].pop( i )
                    new_circ[0].append( i )
                    next_gen.append( new_circ )

            # Evolution
            survivors = []
            for circ in next_gen:
                qfactor.optimize( circ[1], self.utry, min_iters = 0,
                                  diff_tol_r = 1e-6, dist_tol = self.success_threshold )

                dist = qfactor.get_distance( circ[1], self.utry )
                logger.info( "Finished removing gates %s with distance %f"
                             % ( str( circ[0] ), dist ) )

                if dist < self.success_threshold:
                    logger.info( "This circuit survived." )
                    survivors.append( circ )

                    if len( circ[1] ) < len( circuit ):
                        circuit = circ[1]
                    break

            pop = survivors

        return [ Gate( g.utry, g.location ) for g in circuit ]

    def dim_reduce ( self, circuit ):
        logger.info( "Starting Dimension Reduction" )

        # Initialize Population
        pop = [ ( [], circuit ) ]

        while len( pop ) > 0:
            # Asexual Breeding
            logger.info( "Breeding..." )
            next_gen = []
            for circ in pop:
                for i in range( len( circ[1] ) ):
                    new_circ = copy.deepcopy( circ )
                    new_circ[1].pop( i )
                    new_circ[0].append( i )
                    next_gen.append( new_circ )

            # Evolution
            survivors = []
            for circ in next_gen:
                qfactor.optimize( circ[1], self.utry, min_iters = 0,
                                  diff_tol_r = 1e-6, dist_tol = self.success_threshold )

                dist = qfactor.get_distance( circ[1], self.utry )
                logger.info( "Finished removing gates %s with distance %f"
                             % ( str( circ[0] ), dist ) )

                if dist < self.success_threshold:
                    logger.info( "This circuit survived." )
                    survivors.append( circ )

                    if len( circ[1] ) < len( circuit ):
                        circuit = circ[1]

            pop = survivors

        return [ Gate( g.utry, g.location ) for g in circuit ]

