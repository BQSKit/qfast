import logging

import scipy as sp
import numpy as np


logger = logging.getLogger( "qfast" )


def get_num_qubits ( M ):
    """Returns the size of the square matrix, M, in qubits."""

    if not is_square_matrix( M ):
        raise TypeError( "Invalid matrix." )

    return int( np.log2( len( M ) ) )


def is_valid_coupling_graph ( coupling_graph, num_qubits = None ):
    """
    Checks if the coupling graph is valid.

    Args:
        coupling_graph (List[Tuple[int]]): The coupling graph to check.

        num_qubits (int or None): The total number of qubits. All qubits
            should be less than this. If None, don't check.

    Returns:
        (bool): Valid or not
    """

    if not isinstance( coupling_graph, list ):
        logger.debug( "Coupling graph is not a list." )
        return False

    if len( coupling_graph ) == 0:
        return True

    if not all( [ isinstance( pair, tuple ) for pair in coupling_graph ] ):
        logger.debug( "Coupling graph is not a list of tuples." )
        return False

    if not all( [ len( pair ) == 2 for pair in coupling_graph ] ):
        logger.debug( "Coupling graph is not a list of pairs." )
        return False

    if num_qubits is not None:
        if not all( [ qubit < num_qubits
                      for pair in coupling_graph
                      for qubit in pair ] ):
            logger.debug( "Coupling graph has invalid qubits." )
            return False

    if len( coupling_graph ) != len( set( coupling_graph ) ):
        logger.debug( "Coupling graph has duplicates." )
        return False

    if not all( [ len( pair ) == len( set( pair ) )
                  for pair in coupling_graph ] ):
        logger.debug( "Coupling graph has an invalid pair." )
        return False

    return True


def is_valid_location ( location, num_qubits = None ):
    """
    Checks if the location is valid.

    Args:
        location (Tuple[int]): The location to check.

        num_qubits (int or None): The total number of qubits. All qubits
            should be less than this. If None, don't check.

    Returns:
        (bool): Valid or not
    """

    if not isinstance( location, tuple ):
        logger.debug( "Location is not a tuple." )
        return False

    if not all( [ isinstance( qubit, int ) for qubit in location ] ):
        logger.debug( "Location is not a tuple of ints." )
        return False

    if len( location ) != len( set( location ) ):
        logger.debug( "Location has duplicates." )
        return False

    if num_qubits is not None:
        if not all( [ qubit < num_qubits for qubit in location ] ):
            logger.debug( "Location has an invalid qubit." )
            return False

    return True


def is_valid_locations ( locations, num_qubits = None, gate_size = None ):
    """
    Checks if locations is valid.

    Args:
        locations (List[Tuple[int]]): The locations to check.

        num_qubits (int or None): The total number of qubits. All qubits
            should be less than this. If None, don't check.

        gate_size (int or None): The expected size of locations.
            If None, then all locations are to be equal in size."

    Returns:
        (bool): Valid or not
    """

    if not isinstance( locations, list ):
        logger.debug( "Locations is not a list." )
        return False

    if len( locations ) == 0:
        logger.debug( "Locations is empty." )
        return True

    if not all( [ is_valid_location( location, num_qubits )
                  for location in locations ] ):
        logger.debug( "Locations is not a list of valid locations." )
        return False

    if gate_size is None:
        gate_size = len( locations[0] )

    if not all( [ len( location ) == gate_size for location in locations ] ):
        logger.debug( "A location has an invalid size." )
        return False

    if len( locations ) != len( set( locations ) ):
        logger.debug( "Locations has duplicates." )
        return False

    return True


def is_matrix ( M ):
    """Checks if M is a matrix."""

    if not isinstance( M, np.ndarray ): 
        logger.debug( "M is not an numpy array." )
        return False

    if len( M.shape ) != 2:
        logger.debug( "M is not an 2-dimensional array." )
        return False

    return True


def is_square_matrix ( M ):
    """Checks if M is a square matrix."""

    if not is_matrix( M ):
        return False

    if M.shape[0] != M.shape[1]:
        return False

    return True


def is_unitary ( U, tol = 1e-15 ):
    """Checks if U is a unitary matrix."""

    if not is_square_matrix( U ):
        return False

    X = U @ U.conj().T
    Y = U.conj().T @ U
    I = np.identity( X.shape[0] )

    if not np.allclose( X, I, rtol = 0, atol = tol ):
        if logger.isEnabledFor( logging.DEBUG ):
            norm = np.linalg.norm( X - I )
            logger.debug( "Failed unitary condition, ||UU^d - I|| = %e" % norm )
        return False
    
    if not np.allclose( Y, I, rtol = 0, atol = tol ):
        if logger.isEnabledFor( logging.DEBUG ):
            norm = np.linalg.norm( Y - I )
            logger.debug( "Failed unitary condition, ||U^dU - I|| = %e" % norm )
        return False
    
    return True


def is_hermitian ( H, tol = 1e-15 ):
    """Checks if H is a hermitian matrix."""

    if not is_square_matrix( H ):
        return False

    if not np.allclose( H, H.conj().T, rtol = 0, atol = tol ):
        if logger.isEnabledFor( logging.DEBUG ):
            norm = np.linalg.norm( H - H.conj().T )
            logger.debug( "Failed hermitian condition, ||H - H^d|| = %e"
                          % norm )
        return False
    
    return True


def is_skew_hermitian ( H, tol = 1e-15 ):
    """Checks if H is a skew hermitian matrix."""

    if not is_square_matrix( H ):
        return False

    if not np.allclose( -H, H.conj().T, rtol = 0, atol = tol ):
        if logger.isEnabledFor( logging.DEBUG ):
            norm = np.linalg.norm( -H - H.conj().T )
            logger.debug( "Failed skew hermitian condition, ||H - H^d|| = %e"
                          % norm )
        return False
    
    return True


def dot_product ( alpha, sigma ):
    """
    Computes the standard dot product of alpha and sigma.

    Args:
        alpha (List[np.ndarray or int or float]): The alpha values

        sigma (List[np.ndarray or int or float]): The sigma values

    Returns:
        (np.ndarray): Sum of element-wise multiplication of alpha and sigma.

    Raises:
        ValueError: If alpha and sigma are incompatible.
    """

    if len( alpha ) != len( sigma ):
        raise ValueError( "Length of alpha and sigma must be the same." )

    return np.sum( [ a*s for a, s in zip( alpha, sigma ) ], 0 )


def dexpmv ( M, dM ):
    """
    Computes the Matrix exponential F = e^M and its derivative dF.

    User must provide M and its derivative dM. If the arugment dM is a
    vector of paritals then dF will be the respective partial vector.
    This is done using a Pade Approximat with scaling and squaring.
    
    Brančík, Lubomír. "Matlab programs for matrix exponential function
    derivative evaluation." Proc. of Technical Computing Prague 2008
    (2008): 17-24.

    Args:
        M (np.ndarray): Matrix to exponentiate.

        dM (np.ndarray): Derivative(s) of M.

    Returns:
        F (np.ndarray): Exponentiated matrix, i.e. e^M.

        dF (np.ndarray): Derivative(s) of F.
    """

    e = np.log2( np.linalg.norm( M, np.inf ) )
    r = int( max( 0, e + 1 ) ) 
    M = M / ( 2 ** r )
    dM = dM / ( 2 ** r )
    X = M
    Y = dM
    c = 0.5
    F = np.identity( M.shape[0] ) + c*M
    D = np.identity( M.shape[0] ) - c*M
    dF = c*dM
    dD = -c*dM
    q = 6
    p = True
    for k in range( 2, q + 1 ):
        c = c * ( q - k + 1 ) / ( k * ( 2 * q - k + 1 ) )
        Y = dM @ X + M @ Y
        X = M @ X
        cX = c * X
        cY = c * Y
        F = F + cX
        dF = dF + cY
        if p:
            D = D + cX
            dD = dD + cY
        else:
            D = D - cX
            dD = dD - cY
        p = not p
    Dinv = np.linalg.inv( D )
    F = Dinv @ F
    dF = Dinv @ (dF - dD @ F )

    for k in range( 1, r + 1 ):
        dF = dF @ F + F @ dF
        F = F @ F

    return F, dF


def softmax ( x, beta = 20 ):
    """
    Computes the softmax of vector x.

    Args:
        x (np.ndarray): Input vector to softmax.

        beta (int): Beta cofficient to scale steepness of softmax.

    Retrurns:
        (np.ndarray): Output vector of softmax.
    """

    shiftx = beta * ( x - np.max( x ) )
    exps = np.exp( shiftx )
    return exps / np.sum(exps)


def closest_unitary ( A ):
    """
    Calculate the closest unitary to a given matrix.


    Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A.

    D.M.Reich. “Characterisation and Identification of Unitary Dynamics
    Maps in Terms of Their Action on Density Matrices”

    Args:
        A (np.ndarray): The matrix input.

    Returns:
        (np.ndarray): The unitary matrix closest to A.
        Return U as a numpy matrix.
    """

    if not is_square_matrix( A ):
        raise TypeError( "A must be a square matrix." )

    V, __, Wh = sp.linalg.svd( A )
    return V @ Wh

