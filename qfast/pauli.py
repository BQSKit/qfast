"""
This module implements a Pauli Operator Library.

The QFAST Compiler relies on pauli operators tremendously.
All the necessary objects and functions are implemented in this library.
"""

import scipy
import numpy     as np
import itertools as it

# The Pauli Matrices
X = np.array( [ [ 0, 1 ],
                [ 1, 0 ] ], dtype = np.complex128 )

Y = np.array( [ [ 0, -1j ],
                [ 1j,  0 ] ], dtype = np.complex128 )

Z = np.array( [ [ 1,  0 ],
                [ 0, -1 ] ], dtype = np.complex128 )

I = np.array( [ [ 1, 0 ],
                [ 0, 1 ] ], dtype = np.complex128 )


def get_norder_paulis ( n ):
    """
    Recursively constructs the nth-order tensor product of the Pauli group.

    Args:
        n (int): Power of the tensor product of the Pauli group.

    Returns:
        (list of np.array): nth-order Pauli matrices
    """

    if n < 0:
        raise ValueError( "n must be a dimension greater than or equal to 0." )

    if n == 0:
        return [ I ]

    if n == 1:
        return [ I, X, Y, Z ]

    norder_paulis = []
    for pauli_n_1, pauli_1 in it.product( get_norder_paulis( n - 1 ),
                                          get_norder_paulis(1) ):
        norder_paulis.append( np.kron( pauli_n_1, pauli_1 ) )

    return norder_paulis


def get_pauli_n_qubit_projection ( n, q_list, without_identity = False ):
    """
    Returns the nth-order Pauli matrices that act only on qubits in q_list.

    Args:
        n (int): Power of the tensor product of the Pauli group

        q_list (List of int): List of qubit indices

    Returns:
        (list of np.array): nth-order Pauli matrices acting
                            only on qubits in q_list.
    """
    if any( [ q < 0 or q >= n for q in q_list ] ):
        raise ValueError( "Qubit indices must be in [0, n).")

    if len( q_list ) != len( set( q_list ) ):
        raise ValueError( "Qubit indices cannot be duplicates." )

    if len( q_list ) == 0:
        raise ValueError( "Need atleast one qubit index." )

    paulis = get_norder_paulis( n )

    # Nth Order Pauli Matrices can be thought of base 4 number
    # I = 0, X = 1, Y = 2, Z = 3
    # XXY = 1 * 4^2 + 1 & 4^1 + 2 * 4^0 = 22 (base 10)
    # This gives the idx of XXY in paulis
    # Note we read qubit index from the left,
    # so XII corresponds to q = 0
    pauli_n_qubit = []
    for ps in it.product( [ 0, 1, 2, 3 ], repeat = len( q_list ) ):
        idx = 0
        for p, q in zip( ps, q_list ):
            idx += p * ( 4 ** ( n - q - 1 ) )
        pauli_n_qubit.append( paulis[ idx ] )
    if without_identity:
        pauli_n_qubit = pauli_n_qubit[1:]
    return pauli_n_qubit


def pauli_dot_product ( alpha, sigma ):
    """
    Computes the dot product of alpha and sigma.

    Args:
        alpha (list of real numbers) The alpha coefficients

        sigma (list of np.array) The Pauli Matrices

    Returns:
        H (np.array) Hermitian matrix computed from the dot
                     product of alpha and sigma.
    """

    if len( alpha ) != len( sigma ):
        raise ValueError( "Length of alpha and sigma must be the same." )

    return np.sum( [ a*s for a, s in zip( alpha, sigma ) ], 0 )
