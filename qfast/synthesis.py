"""
This module implements the main synthesize function.
"""

import tensorflow as tf
import numpy      as np

from .singlegatelayer import SingleGateLayer
from .multigatelayer import MultiGateLayer


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
        return np.sqrt( 1 - ( num / dem ) )
    else:
        mat = tf.matmul( tf.transpose( tf.conj( X ) ), Y )
        num = tf.abs( tf.trace( mat ) ) ** 2
        dem = int( mat.shape[0] ) ** 2
        return tf.sqrt( 1 - ( num / dem ) )


def fixed_size_synthesize ( target, layer_count, gate_size, verbosity,
                            init_values = None ):
    """
    Attempts to synthesize the target unitary with a fixed number
    of gates of size gate_size.

    Args:
        target (np.ndarray): Target unitary

        layer_count (int): The number of gates to synthesize

        gate_size (int): The size of each gate

        verbosity (int): If 2 will print loss values

    Returns:
        Circuit (None or List[Tuple[Tuple[int], List[floats]]]):
            Returns None if plateau'd before could find a solution;
            returns a circuit which maps gate_link to gate_params as
            a Pauli Expansion.
    """

    num_qubits = int( np.log2( len( target ) ) )

    tf.reset_default_graph()

    if init_values == None:
        init_values = [ None ] * layer_count

    while len( init_values ) < layer_count:
        init_values.append( None )

    layers = [ MultiGateLayer( "Layer%d" % i, num_qubits,
                               gate_size = gate_size,
                               parity = (i + 1) % 2,
                               init_values = init_values[i] )
               for i in range( layer_count ) ]

    tensor = layers[0].get_unitary_tensor()
    for layer in layers[1:]:
        tensor = tf.matmul( layer.get_unitary_tensor(), tensor )

    loss_fn   = hilbert_schmidt_distance( target, tensor )
    optimizer = tf.train.AdamOptimizer( 0.01 )
    train_op  = optimizer.minimize( loss_fn )
    init_op   = tf.global_variables_initializer()

    loss_values = []

    with tf.Session() as sess:
        sess.run( init_op )

        while ( True ):
            for i in range( 20 ):
                loss = sess.run( [ train_op, loss_fn ] )[1]

            if loss < 0.02:

                if verbosity >= 1:
                    print( "Found circuit with loss: %f" % loss )

                circuit = []
                for l in layers:
                    circuit.append( ( l.get_link( sess ), l.get_gate_vals( sess ) ) )
                return ( True, circuit )

            loss_values.append( loss )

            if verbosity >= 2:
                print( loss )

            if len( loss_values ) > 100:
                min_value = np.min( loss_values[-100:] )
                max_value = np.max( loss_values[-100:] )

                # Plateau Detected
                if max_value - min_value < 0.001:
                    return ( False, [ l.get_values( sess ) for l in layers ] )

            if len( loss_values ) > 1000:
                min_value = np.min( loss_values[-1000:] )
                max_value = np.max( loss_values[-1000:] )

                # Plateau Detected
                if max_value - min_value < 0.02:
                    return ( False, [ l.get_values( sess ) for l in layers ] )


def synthesize ( target, start_layer, layer_step, gate_size, verbosity = 0 ):
    """
    Synthesizes a circuit that implements the target unitary.

    Args:
        target (np.ndarray): Target unitary

        start_layer (int): The number of gates to start searching from

        layer_step (int): The number of added gates in each step

        gate_size (int): The size of each gate

        verbosity (int): If 2 will print loss values, +
                            1 will print state changes

    Returns:
        Circuit (List[Tuple[Tuple[int], List[floats]]]):
            A circuit which maps gate_link to gate_params as a Pauli Expansion.
    """

    layer_count = start_layer
    init_values = [ None ] * layer_count

    print( "Starting to search with %d layers" % layer_count )

    # Stride search over layer_count
    while ( True ):
        result = fixed_size_synthesize( target, layer_count, gate_size,
                                        verbosity, init_values )

        if result[0]:

            if verbosity >= 1:
                print( "Found a circuit with %d layers" % layer_count )

            circuit = result[1]
            break

        init_values  = result[1]
        layer_count += layer_step

        if verbosity >= 1:
            print( "Added a layer, now: %d layers" % layer_count )

    found_circuit = circuit

    # Search backwards for a shorter circuit
    for i in range( layer_step - 1 ):
        layer_count -= 1

        if verbosity >= 1:
            print( "Removing a layer, now: %d layers" % layer_count )

        result = fixed_size_synthesize( target, layer_count, gate_size,
                                        verbosity, init_values )

        if result[0]:

            if verbosity >= 1:
                print( "Found a circuit with %d layers" % layer_count )

            found_circuit = result[1]

        elif circuit is None:
            break

    return found_circuit


def refine_circuit ( target, circuit, verbosity = 0 ):
    """
    Refines circuit's precision by using SingleGateLayers instead
    of MultiGateLayers.

    Args:
        target (np.ndarray): Target unitary

        circuit (List[Tuple[Tuple[int], List[floats]]]): Circuit to be refined

        verbosity (int): If 2 will print loss values

    Returns:
        Circuit (List[Tuple[Tuple[int], List[floats]]]): Refined circuit
    """

    num_qubits  = int( np.log2( len( target ) ) )
    layer_count = len( circuit )
    gate_size   = len( circuit[0][0] )

    tf.reset_default_graph()

    layers = [ SingleGateLayer( "Layer%d" % i, num_qubits,
                                link = circuit[i][0],
                                init_values = circuit[i][1],
                                gate_size = gate_size )
               for i in range( layer_count ) ]

    tensor = layers[0].get_unitary_tensor()
    for layer in layers[1:]:
        tensor = tf.matmul( layer.get_unitary_tensor(), tensor )

    loss_fn   = hilbert_schmidt_distance( target, tensor )
    optimizer = tf.train.AdamOptimizer( 1e-6 )
    train_op  = optimizer.minimize( loss_fn )
    init_op   = tf.global_variables_initializer()

    loss_values = []

    with tf.Session() as sess:
        sess.run( init_op )

        for i in range( 10000 ):
            for j in range( 50 ):
                loss = sess.run( [ train_op, loss_fn ] )[1]

            if loss < 1e-7:
                break

            loss_values.append( loss )

            if verbosity >= 2:
                print( loss )

            if len( loss_values ) > 100:
                min_value = np.min( loss_values[-100:] )
                max_value = np.max( loss_values[-100:] )

                # Plateau Detected
                if max_value - min_value < 1e-5:
                    break

            if len( loss_values ) > 500:
                min_value = np.min( loss_values[-500:] )
                max_value = np.max( loss_values[-500:] )

                # Plateau Detected
                if max_value - min_value < 1e-3:
                    break

        if verbosity >= 1:
            print( "Refined circuit to %f error" % loss )

        new_circuit = []
        for l in layers:
            new_circuit.append( ( l.get_link( sess ), l.get_gate_vals( sess ) ) )
        return new_circuit
