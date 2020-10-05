"""This module implements pauli operations."""

import scipy
import numpy      as np
import itertools  as it

# The Pauli Matrices
X = np.array( [ [ 0, 1 ],
                [ 1, 0 ] ], dtype = np.complex128 )

Y = np.array( [ [ 0, -1j ],
                [ 1j,  0 ] ], dtype = np.complex128 )

Z = np.array( [ [ 1,  0 ],
                [ 0, -1 ] ], dtype = np.complex128 )

I = np.array( [ [ 1, 0 ],
                [ 0, 1 ] ], dtype = np.complex128 )


# Pauli Cache
_norder_paulis_map = [ np.array( [ I ] ), np.array( [ I, X, Y, Z ] ) ]


def get_norder_paulis ( n ):
    """
    Recursively constructs the nth-order tensor product of the Pauli group.

    Args:
        n (int): Power of the tensor product of the Pauli group.

    Returns:
        (np.ndarray): nth-order Pauli matrices

    Raises:
        ValueError: If n is less than 0.
    """

    if n < 0:
        raise ValueError( "n must be nonnegative" )

    if len( _norder_paulis_map ) > n:
        return _norder_paulis_map[n]

    norder_paulis = []
    for pauli_n_1, pauli_1 in it.product( get_norder_paulis( n - 1 ),
                                          get_norder_paulis(1) ):
        norder_paulis.append( np.kron( pauli_n_1, pauli_1 ) )

    _norder_paulis_map.append( np.array( norder_paulis ) )

    return _norder_paulis_map[n]


def get_pauli_n_qubit_projection ( n, q_set ):
    """
    Returns the nth-order Pauli matrices that act only on qubits in q_set.

    Args:
        n (int): Power of the tensor product of the Pauli group

        q_set (Tuple[int] or Set[int]): Qubit indices

    Returns:
        pauli_n_qubit (np.ndarray): nth-order Pauli matrices acting
                                        only on qubits in q_set.

    Raises:
        ValueError: if q_set is an invalid set of qubit indicies.
    """

    if any( [ q < 0 or q >= n for q in q_set ] ):
        raise ValueError( "Qubit indices must be in [0, n).")

    if len( q_set ) != len( set( q_set ) ):
        raise ValueError( "Qubit indices cannot have duplicates." )

    if len( q_set ) == 0:
        raise ValueError( "Need atleast one qubit index." )

    paulis = get_norder_paulis( n )

    # Nth Order Pauli Matrices can be thought of base 4 number
    # I = 0, X = 1, Y = 2, Z = 3
    # XXY = 1 * 4^2 + 1 & 4^1 + 2 * 4^0 = 22 (base 10)
    # This gives the idx of XXY in paulis
    # Note we read qubit index from the left,
    # so X in XII corresponds to q = 0
    pauli_n_qubit = []
    for ps in it.product( [ 0, 1, 2, 3 ], repeat = len( q_set ) ):
        idx = 0
        for p, q in zip( ps, q_set ):
            idx += p * ( 4 ** ( n - q - 1 ) )
        pauli_n_qubit.append( paulis[ idx ] )

    return np.array( pauli_n_qubit )


def unitary_log_no_i ( U, tol = 1e-15 ):
    """
    Solves for H in U = e^{iH}

    Args:
        U (np.ndarray): The unitary to decompose

    Returns:
        H (np.ndarray): e^{iH} = U
    """

    if not utils.is_unitary( U, tol ):
        raise TypeError( "Input is not unitary." )

    T, Z = la.schur( U )
    T = np.diag( T )
    D = T / np.abs( T )
    D = np.diag( np.log( D ) )
    H0 = -1j * (Z @ D @ Z.conj().T)
    return 0.5 * H0 + 0.5 * H0.conj().T


def pauli_expansion ( H, tol = 1e-15 ):
    """
    Computes a Pauli expansion of the hermitian matrix H.

    Args:
        H (np.ndarray): The hermitian matrix

    Returns:
        X (list of floats): The coefficients of a Pauli expansion for H,
                            i.e., X dot Sigma = H where Sigma is
                            Pauli matrices of same size of H
    """

    if not utils.is_hermitian( H, tol ):
        raise ValueError( "H must be hermitian." )

    # Change basis of H to Pauli Basis (solve for coefficients -> X)
    n = int( np.log2( len( H ) ) )
    paulis = get_norder_paulis( n )
    flatten_paulis = [ np.reshape( pauli, 4 ** n ) for pauli in paulis ]
    flatten_H = np.reshape( H, 4 ** n )
    A = np.stack( flatten_paulis, axis = -1 )
    X = np.real( np.matmul( np.linalg.inv( A ), flatten_H ) )
    return X

