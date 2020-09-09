import numpy as np
from functools import reduce

from qfast.circuitmodel import CircuitModel
from qfast.gate import Gate
from .genericgate import GenericGate
from .fixedgate import FixedGate

class PermModel ( CircuitModel ):

    def __init__ ( self, utry, gate_size, locations, optimizer ):
        super().__init__( utry, gate_size, locations, optimizer )
        self.head = GenericGate( self.num_qubits, self.gate_size, self.locations )
        self.gates = [ self.head ]
        self.param_ranges = [0, self.head.get_param_count()]

        self.success_threshold = 5e-5
        self.progress_threshold = 5e-3

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
        num = np.abs( np.trace( self.utry_dag @ M ) ) ** 2
        dem = M.shape[0] ** 2
        return np.sqrt( 1 - ( num / dem ) )

    def success ( self, distance ):
        return distance < self.success_threshold

    def progress ( self, distance, last_dist ):
        return last_dist - distance > self.progress_threshold

    def expand ( self, location ):
        new_gate = FixedGate( self.num_qubits, self.gate_size, location )
        self.gates.insert( -1, new_gate )

        num_params = new_gate.get_param_count()
        self.param_ranges.insert( -1, self.param_ranges[-2] + num_params )
        self.param_ranges[-1] += num_params

        self.head.lift_restrictions()

    def finalize ( self, location, xin ):
        bound = self.param_ranges[ -2 ] 
        xhead = self.head.convert_input_to_fixed( xin[ bound : ] )
        xin = np.concatenate( ( xin[ : bound ], xhead ) )

        self.gates.pop( -1 )
        self.param_ranges.pop( -1 )

        new_gate = FixedGate( self.num_qubits, self.gate_size, location )
        num_params = new_gate.get_param_count()
        self.param_ranges.append( self.param_ranges[-1] + num_params )
        self.gates.append( new_gate )

        xout = self.optimizer.minimize_fine( self.objective_fn, xin )

        distance = self.distance( xout )

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

    def solve ( self ):
        failed_locs = []
        last_dist = 1
        depth = 1
        xout = None

        while True:
            
            # if len( saved_locs ) == 0:
                # tp = t.copy()
            # else:
                # tp = t.copy()
                # tp.remove( saved_locs[-1] )

            # for loc in failed_locs:
                # tp.remove( loc[0] )

            # for j in range( i ):
                # circ.append_gate( FixedGate( n, m, saved_locs[j] ) )
            # circ.append_gate( GenericGate( n, m, tp ) )

            xin = self.get_initial_input()
            # print ( self.param_ranges )

            # if xout is not None:
                # print( len( xin ) )
                # print( len( xout ) )
                # xin[ -len( xout ) : ] = xout
            if last_dist <= 1e-3:
                xout = self.optimizer.minimize_fine( self.objective_fn, xin )
            else:
                xout = self.optimizer.minimize_coarse( self.objective_fn, xin )

            distance = self.distance( xout )

            chosen_location = self.head.get_chosen_location( xout )
            print( distance, chosen_location )

            if self.success( distance ):
                print( "SUCCESS" )
                return self.finalize( chosen_location, xout )


            if self.progress( distance, last_dist ):
                print( "PROGRESS" )
                last_dist = distance
                self.expand( chosen_location )
                self.head.restrict( chosen_location )
                depth += 1

            elif self.head.cannot_restrict():
                print( "CANNOT RESTRICT" )
                failed_locs.sort( key = lambda x : x[1] )
                chosen_location, last_dist = failed_locs[0]
                self.expand( chosen_location )
                depth += 1
                failed_locs = []

            else:
                print( "RESTRICT" )
                failed_locs.append( ( chosen_location, distance ) )
                self.head.restrict( chosen_location )

            # print( "-------------------------------------" )
            # print( "-------------------------------------" )
            # print( "RESULTS FOR i = ", i )
            # print( "-------------------------------------" )
            # print( res.fun )
            # loss = 1 + (res.fun / (2**n))
            # print( loss )
            # print( res.x )
            # print( res.message )
            # print( utils.softmax( res.x[ -len( tp ) : ], beta = 10 ) )
            # if last_dist - distance <= self.progress_threshold:
                # print( "Didn't improve." )
                # depth -= 1
                # failed_locs.append( ( tp[ np.argmax( res.x[ -len( tp ) : ] ) ], loss ) )
            # else:
                # saved_locs.append( tp[ np.argmax( res.x[ -len( tp ) : ] ) ] )
                # failed_locs = []
                # last_loss = loss

            # if len( failed_locs ) == len( t ) - 1:
                # min_loss = 1
                # min_loc = None
                # for loc, floss in failed_locs:
                    # if floss < min_loss:
                        # min_loss = floss
                        # min_loc = loc
                # failed_locs = []
                # saved_locs.append( min_loc )
                # last_loss = min_loss
                # i += 1
                # print( "Ugh, going on anyways." )

            # if loss < 1e-3:
                # print( "FOUND CIRCUIT!" )
                # i += 10000
            # print( saved_locs )
            # print( failed_locs )
            # i += 1
            # print( "-------------------------------------" )
            # print( "-------------------------------------" )


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

