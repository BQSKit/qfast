import numpy as np
from functools import reduce

from qfast.decomposition.circuitmodel import CircuitModel
from qfast.gate import Gate
from .fixedgate import FixedGate

import logging
logger = logging.getLogger( "qfast" )

class FixedModel ( CircuitModel ):

    def __init__ ( self, utry, gate_size, locations, optimizer, structure ):
        super().__init__( utry, gate_size, locations, optimizer )

        self.gates = []
        self.param_ranges = [ 0 ]
        for location in structure:
            self.gates.append( FixedGate( self.num_qubits, self.gate_size, location ) )
            self.param_ranges.append( self.param_ranges[-1] + self.gates[-1].get_param_count() )

        self.success_threshold = 1e-3

    def get_initial_input ( self ):
        return np.concatenate( [ gate.get_initial_input() for gate in self.gates ] )

    def objective_fn ( self, x ):
        M, dM = self.get_matrix_and_derivatives( x )
        obj = -np.real( np.trace( self.utry_dag @ M ) )
        jacs = []
        for dm in dM:
            jacs.append( -np.real( np.trace( self.utry_dag @ dm ) ) )
        jacs = np.array( jacs )
        return obj, jacs

    def distance ( self, x ):
        M = self.get_matrix( x )
        num = np.abs( np.trace( self.utry_dag @ M ) )
        dem = M.shape[0]
        return 1 - ( num / dem )

    def success ( self, distance ):
        return distance < self.success_threshold

    def solve ( self ):
        xin = self.get_initial_input()

        xout = self.optimizer.minimize_fine( self.objective_fn, xin )

        distance = self.distance( xout )

        logger.info( f"Completed at distance: {distance}" )

        return self.get_gate_list( xout )

    def get_gate_list ( self, x ):
        gate_list = []

        for i, gate in enumerate( self.gates ):
            lower_bound = self.param_ranges[ i ]
            upper_bound = self.param_ranges[ i + 1 ]
            M = gate.get_actual_matrix( x[ lower_bound : upper_bound ] )
            L = gate.get_location()
            gate_list.append( Gate( M, L ) )

        return gate_list

    def get_input_slice ( self, x, gate_idx ):
        if gate_idx < 0:
            lower_bound = self.param_ranges[ i - 1 ]
            upper_bound = self.param_ranges[ i ]
        else:
            lower_bound = self.param_ranges[ i ]
            upper_bound = self.param_ranges[ i + 1 ]

        return x[ lower_bound : upper_bound ]

    def get_param_count ( self ):
        return self.param_ranges[-1]

    def get_matrix ( self, x ):
        if len( self.gates ) == 0:
            return np.identity( self.utry_dag.shape[0] )
        
        if len( self.gates ) == 1:
            return self.gates[0].get_matrix(x)

        matrices = []

        for i, gate in enumerate( self.gates ):
            lower_bound = self.param_ranges[ i ]
            upper_bound = self.param_ranges[ i + 1 ]
            matrices.append( gate.get_matrix( x[ lower_bound : upper_bound ] ) )

        return reduce( np.matmul, reversed( matrices ) )

    def get_matrix_and_derivatives ( self, x ):
        if len( self.gates ) == 0:
            return np.identity( self.utry_dag.shape[0] ), np.array([])
        
        if len( self.gates ) == 1:
            return self.gates[0].get_matrix_and_derivatives(x)

        matrices = []
        derivatives = []

        for i, gate in enumerate( self.gates ):
            lower_bound = self.param_ranges[ i ]
            upper_bound = self.param_ranges[ i + 1 ]
            M, J = gate.get_matrix_and_derivatives( x[ lower_bound : upper_bound ] )
            matrices.append( M )
            derivatives.append( J )

        matrix = reduce( np.matmul, reversed( matrices ) )
        jacs = []

        for i, dM in enumerate( derivatives ):

            if i + 1 < len( derivatives ):
                left = reduce( np.matmul, reversed( matrices[i+1:] ) )
            else:
                left = np.identity( self.utry_dag.shape[0] )


            if i != 0:
                right = reduce( np.matmul, reversed( matrices[:i] ) )
            else:
                right = np.identity( self.utry_dag.shape[0] )

            for dm in dM:
                jacs.append( left @ dm @ right )

        return matrix, np.array( jacs )

