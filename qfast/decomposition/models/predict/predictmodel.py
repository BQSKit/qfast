import numpy as np
import scipy.linalg as la
import itertools as it
from qfast.gate import Gate
from qfast.pauli import get_norder_paulis, get_pauli_n_qubit_projection
from qfast.utils import dot_product, closest_unitary, is_unitary
from qfast.decomposition.circuitmodel import CircuitModel
from .fixedgate import FixedGate
import scipy.optimize as opt
from functools import reduce

def unitary_log_no_i ( U ):
    """
    Solves for H in U = e^{iH}

    Args:
        U (np.ndarray): The unitary to decompose

    Returns:
        H (np.ndarray): e^{iH} = U
    """

    if not is_unitary( U ):
        U = closest_unitary( U )

    T, Z = la.schur( U )
    T = np.diag( T )
    D = T / np.abs( T )
    D = np.diag( np.log( D ) )
    H0 = -1j * (Z @ D @ Z.conj().T)
    return 0.5 * H0 + 0.5 * H0.conj().T


def pauli_expansion ( H ):
    """
    Computes a Pauli expansion of the hermitian matrix H.

    Args:
        H (np.ndarray): The hermitian matrix

    Returns:
        X (list of floats): The coefficients of a Pauli expansion for H,
                            i.e., X dot Sigma = H where Sigma is
                            Pauli matrices of same size of H
    """

    if not np.allclose( H, H.conj().T, rtol = 0, atol = 1e-15 ):
        raise ValueError( "H must be hermitian." )

    # Change basis of H to Pauli Basis (solve for coefficients -> X)
    n = int( np.log2( len( H ) ) )
    paulis = get_norder_paulis( n )
    flatten_paulis = [ np.reshape( pauli, 4 ** n ) for pauli in paulis ]
    flatten_H      = np.reshape( H, 4 ** n )
    A = np.stack( flatten_paulis, axis = -1 )
    X = np.real( np.matmul( np.linalg.inv( A ), flatten_H ) )
    return X
"""
n = 3
#utry = np.loadtxt( "qft3.unitary", dtype = np.complex128 )
pauli_in = np.random.random( 4 ** n )
utry = la.expm( 1j * dot_product( pauli_in, get_norder_paulis( n ) ) )
pauli_out = pauli_expansion( unitary_log_no_i_eig( utry ) )
#print( np.allclose( pauli_in, pauli_out ) )
#print( paulis )
#print( np.sqrt( np.sum( np.square( paulis ) ) ) )
mag_in = np.sqrt( np.sum( np.square( pauli_in ) ) )
mag_out = np.sqrt( np.sum( np.square( pauli_out ) ) )
#print( mag_in, mag_out )
#print( np.allclose( pauli_in / mag_in, pauli_out / mag_out, rtol = 0, atol = 1e-5 ) )
"""

def get_pauli_str ( idx, num_qubits ):
    keys = [ "I", "X", "Y", "Z" ]
    string = ""
    for i in range( num_qubits ):
        string = keys[ idx & 3 ] + string
        idx >>= 2
    return string

def pretty_print_paulis ( paulis ):
    num_qubits = int( np.log2( len( paulis ) ) / 2 )
    for i, p in enumerate( paulis ):
        print( get_pauli_str( i, num_qubits ), p )

# pretty_print_paulis( pauli_out )

def count_entanglement ( paulis, q_list ):
    num_qubits = int( np.log2( len( paulis ) ) / 2 )

    sum = 0
    for ps in it.product( [ 1, 2, 3 ], repeat = len( q_list ) ):
        idx = 0
        for p, q in zip( ps, q_list ):
            idx += p * ( 4 ** ( num_qubits - q - 1 ) )
        sum += np.abs( paulis[ idx ] )
        #sum += paulis[ idx ]

    return sum


def get_pauli_projection ( paulis, q_list ):
    num_qubits = int( np.log2( len( paulis ) ) / 2 )
    pauli_project = []
    for ps in it.product( [ 0, 1, 2, 3 ], repeat = len( q_list ) ):
        idx = 0
        for p, q in zip( ps, q_list ):
            idx += p * ( 4 ** ( num_qubits - q - 1 ) )
        pauli_project.append( paulis[ idx ] )
    return pauli_project

def get_most_entangled ( paulis, t ):
    max_val = -1000
    max_l = None

    for l in t:
        val = count_entanglement( paulis, l )
        if val > max_val:
            max_val = val
            max_l = l

    return max_l

def get_entangled_order ( paulis, t ):
    entangle_count = [ (l, count_entanglement( paulis, l )) for l in t ]
    return sorted( entangle_count, key = lambda x : x[1], reverse = True )


class PredictionModel ( CircuitModel ):

    def __init__ ( self, utry, gate_size, locations, optimizer ):
        super().__init__( utry, gate_size, locations, optimizer )
        self.success_threshold = 1e-3
        self.progress_threshold = 5e-3
        #self.gate = GenericGate( self.num_qubits, self.gate_size, self.locations )

    def solve ( self ):
        utrys = []
        self.locs = []
        self.prefix = np.identity( 2**self.num_qubits )
        self.suffix = np.identity( 2**self.num_qubits )
        dists = [1]
        fine = False

        for i in range( 1000 ):
            print( "Depth:", i )
            paulis = pauli_expansion( unitary_log_no_i( self.suffix.conj().T @ self.utry ) )
            l = get_most_entangled( paulis, self.locations )

            self.gate = FixedGate( self.num_qubits, self.gate_size, l )
            #self.gate.lift_restrictions()

            #if len( locs ) > 0:
            #    self.gate.restrict( locs[-1] )

            xin = self.get_initial_input()

            if fine:
                xout = self.optimizer.minimize_fine( self.objective_fn, xin )
            else:
                xout = self.optimizer.minimize_coarse( self.objective_fn, xin )

            #utrys.append( self.gate.get_fixed_matrix( xout ) )
            #locs.append( self.gate.get_chosen_location( xout ) )
            utrys.append( closest_unitary( self.gate.get_matrix( xout ) ) )
            self.suffix = utrys[-1] @ self.suffix
            

            last_dist = 1 - (np.abs( np.trace( self.suffix.conj().T @ self.utry ) ) / (2**self.num_qubits))
            for loop_iter in range( 5 ):
                self.locs = []
                self.xs = []
                for j in range( i + 1 ):
                    print( "Reoptimizing gate at depth:", j )
                    self.suffix = closest_unitary( self.suffix @ utrys[j].conj().T )
                    roll = np.random.randint(1)
                    if roll % 3 == 0:
                        paulis = pauli_expansion( unitary_log_no_i( self.suffix.conj().T @ self.utry @ self.prefix.conj().T ) )
                    elif roll % 3 == 1:
                        paulis = pauli_expansion( unitary_log_no_i( self.utry @ self.prefix.conj().T ) )
                    l = get_most_entangled( paulis, self.locations )
                    entangled_order = get_entangled_order( paulis, self.locations )
                    #print( entangled_order )
                    assert( l == entangled_order[0][0] )

                    if roll % 3 == 2:
                        l = np.random.choice( self.locations, 1 )

                    if len( self.locs ) > 0 and l == self.locs[-1]:
                        l = entangled_order[1][0]

                    self.locs.append( l )
                    self.gate = FixedGate( self.num_qubits, self.gate_size, l )
                    #self.gate.lift_restrictions()

                    #if j - 1 < len( locs ) and j - 1 >= 0:
                    #    self.gate.restrict( locs[ j - 1 ] )

                    #if j + 1 < len( locs ):
                    #    self.gate.restrict( locs[ j + 1 ] )

                    xin = self.get_initial_input()

                    if fine:
                        xout = self.optimizer.minimize_fine( self.objective_fn, xin )
                    else:
                        xout = self.optimizer.minimize_coarse( self.objective_fn, xin )

                    self.xs.append( xout )
                    
                    #utrys[j] = self.gate.get_fixed_matrix( xout )
                    #locs[j] = self.gate.get_chosen_location( xout )
                    utrys[j] = self.gate.get_matrix( xout )
                    self.prefix = utrys[j] @ self.prefix
                self.suffix = self.prefix
                self.prefix = np.identity( 2**self.num_qubits )
                dist = 1 - (np.abs( np.trace( self.suffix.conj().T @ self.utry ) ) / (2**self.num_qubits))
                #print( self.locs )
                if dist < self.success_threshold:
                    self.dist = dist
                    return self.finalize()

                if not fine and dist < 1e-2:
                    fine = True
                    continue

                if last_dist - dist < 1e-3 and dist < dists[-1]:
                    dists.append( dist )
                    break

                last_dist = dist

            print( dist )


    def finalize ( self ):
        self.gates = [ FixedGate( self.num_qubits, self.gate_size, l ) for l in self.locs ]
        self.param_ranges = [0]

        for gate in self.gates:
            self.param_ranges.append( self.param_ranges[-1] + gate.get_param_count() )
        
        xout = None

        if self.dist > self.success_threshold:
            xin = np.concatenate( self.xs )
            xout = self.optimizer.minimize_fine( self.finalize_fn, xin )
            self.dist = self.distance( xout )
            print( self.dist )

        if xout is None:
            xout= np.concatenate( self.xs )

        return self.get_gate_list( xout )

    def get_gate_list ( self, x ):
        gate_list = []

        for i, gate in enumerate( self.gates ):
            lower_bound = self.param_ranges[ i ]
            upper_bound = self.param_ranges[ i + 1 ]
            M = closest_unitary( gate.get_actual_matrix( x[ lower_bound : upper_bound ] ) )
            L = gate.get_location()
            gate_list.append( Gate( M, L ) )

        return gate_list


    def get_initial_input ( self ):
        return self.gate.get_initial_input()

    def finalize_fn ( self, x ):
        M, dM = self.get_matrix_and_derivatives( x )
        obj = -np.real( np.trace( self.utry_dag @ M ) )
        jacs = []
        for dm in dM:
            jacs.append( -np.real( np.trace( self.utry_dag @ dm ) ) )
        jacs = np.array( jacs )
        return obj, jacs
    
    def objective_fn ( self, x ):
        M, dM = self.gate.get_matrix_and_derivatives( x )
        obj = -np.real( np.trace( ( self.prefix @ self.utry_dag @ self.suffix  ) @ M ) )
        jacs = []
        for dm in dM:
            jacs.append( -np.real( np.trace( ( self.prefix @ self.utry_dag @ self.suffix ) @ dm ) ) )
        jacs = np.array( jacs )
        return obj, jacs

    """
    def objective_fn ( self, x ):
        M, dM = self.gate.get_matrix_and_derivatives( x )
        trace = np.trace( ( self.prefix @ self.utry_dag @ self.suffix  ) @ M )
        obj = -np.abs( trace )
        jacs = []
        for dm in dM:
            jacs.append( np.trace( ( self.prefix @ self.utry_dag @ self.suffix ) @ dm ) )
        jacs = np.array( jacs )

        jacs = (np.real( trace ) * np.real( jacs )) + (np.imag( trace ) * np.imag( jacs ))
        jacs = jacs / obj

        return obj, jacs
    """
        
    def distance ( self, x ):
        M,_ = self.get_matrix_and_derivatives( x )
        num = np.abs( np.trace( self.utry_dag @ M ) )
        dem = M.shape[0]
        return 1 - ( num / dem )

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
