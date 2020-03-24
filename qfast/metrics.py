"""
This module implements unitary distance functions.
"""

import tensorflow as tf
import numpy      as np


def hilbert_schmidt_distance ( X, Y ):
    """
    Calculates a distance based on the Hilbert Schmidt inner product.

    Args:
        X: First Operator
        Y: Second Operator

    Returns:
        Either a result in numpy or a tensor in tensorflow.
    """

    if not isinstance( X, np.ndarray ) and not isinstance( X, tf.Tensor ):
        raise TypeError( "X must be either a np.ndarray or a tf.Tensor." )

    if not isinstance( Y, np.ndarray ) and not isinstance( Y, tf.Tensor ):
        raise TypeError( "Y must be either a np.ndarray or a tf.Tensor." )

    if X.shape != Y.shape:
        raise ValueError( "X and Y must have same shape." )

    if isinstance( X, np.ndarray ) and isinstance( Y, np.ndarray ):
        mat = np.matmul( np.transpose( np.conj( X ) ), Y )
        num = np.abs( np.trace( mat ) ) ** 2
        dem = mat.shape[0] ** 2

        quotient = num / dem
        if np.allclose( 1, quotient, rtol = 0, atol = 1e-15 ):
            quotient = 1

        return np.sqrt( 1 - quotient )
    else:
        mat = tf.matmul( tf.transpose( tf.conj( X ) ), Y )
        num = tf.abs( tf.trace( mat ) ) ** 2
        dem = int( mat.shape[0] ) ** 2
        return tf.sqrt( 1 - ( num / dem ) )