"""
This module implements the main decomposition functions.
"""

import tensorflow as tf
import numpy      as np

from decomposition import FixedGate, GenericGate
from tools import hilbert_schmidt_distance


def fixed_depth_exploration ( target, num_qubits, gate_size, gate_fun_vals,
                              gate_loc_vals, exploration_distance = 0.01,
                              learning_rate = 0.01 ):
    """
    Attempts to synthesize the target unitary with a fixed number
    of gates of size gate_size.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        gate_fun_vals (List[List[float]]): Gate function values

        gate_loc_vals (List[List[float]]): Gate location values

        exploration_distance (float): Exploration's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (Tuple[bool, List[List[float]], List[List[float]]]):
            True if succeeded in hitting exploration distance
            and the final gate_fun_vals and gate_loc_vals
    """

    tf.reset_default_graph()

    layers = [ GenericGate( "Gate%d" % i, num_qubits, gate_size,
                            gate_fun_vals[i], gate_loc_vals[i],
                            parity = (i + 1) % 2 )
               for i in range( len( gate_fun_vals ) ) ]

    tensor = layers[0].get_tensor()
    for layer in layers[1:]:
        tensor = tf.matmul( layer.get_tensor(), tensor )

    loss_fn   = hilbert_schmidt_distance( target, tensor )
    optimizer = tf.train.AdamOptimizer( learning_rate )
    train_op  = optimizer.minimize( loss_fn )
    init_op   = tf.global_variables_initializer()

    loss_values = []

    with tf.Session() as sess:
        sess.run( init_op )

        while ( True ):
            for i in range( 20 ):
                loss = sess.run( [ train_op, loss_fn ] )[1]

            if loss < exploration_distance:

                # if verbosity >= 1:
                #     print( "Found circuit with loss: %f" % loss )

                return ( True,
                         [ l.get_fun_vals( sess ) for l in layers ],
                         [ l.get_loc_vals( sess ) for l in layers ] )

            loss_values.append( loss )

            # if verbosity >= 2:
            #     print( loss )

            if len( loss_values ) > 100:
                min_value = np.min( loss_values[-100:] )
                max_value = np.max( loss_values[-100:] )

                # Plateau Detected
                if max_value - min_value < learning_rate / 10:
                    return ( False,
                             [ l.get_fun_vals( sess ) for l in layers ],
                             [ l.get_loc_vals( sess ) for l in layers ] )

            if len( loss_values ) > 500:
                min_value = np.min( loss_values[-500:] )
                max_value = np.max( loss_values[-500:] )

                # Plateau Detected
                if max_value - min_value < learning_rate:
                    return ( False,
                             [ l.get_fun_vals( sess ) for l in layers ],
                             [ l.get_loc_vals( sess ) for l in layers ] )


def exploration ( target, num_qubits, gate_size, start_depth, depth_step,
                  exploration_distance = 0.01, learning_rate = 0.01 ):
    """
    Synthesizes a circuit that implements the target unitary.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        start_depth (int): The number of gates to start searching from

        depth_step (int): The number of added gates in each step

        exploration_distance (float): Exploration's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (Tuple[List[List[float]], List[List[float]]]):
            The final gate_fun_vals and gate_loc_vals
    """

    depth = start_depth
    gate_fun_vals = [ None ] * depth
    gate_loc_vals  = [ None ] * depth

    print( "Starting to search with %d layers" % layer_count )

    # Stride search over layer_count
    while ( True ):
        result = fixed_depth_exploration( target, num_qubits, gate_size,
                                          gate_fun_vals, gate_loc_vals,
                                          exploration_distance, learning_rate )

        success, gate_fun_vals, gate_loc_vals = result

        if success:

            # if verbosity >= 1:
            #     print( "Found a circuit with %d layers" % layer_count )

            # circuit = result[1]
            break

        gate_fun_vals += [ None ] * depth_step
        gate_loc_vals  += [ None ] * depth_step
        depth += depth_step

        # if verbosity >= 1:
        #     print( "Added a layer, now: %d layers" % layer_count )

    # Remember good results
    found_gate_fun_vals = gate_fun_vals
    found_gate_loc_vals  = gate_loc_vals

    # Search backwards for a shorter circuit
    for i in range( depth_step - 1 ):
        depth -= 1

        # if verbosity >= 1:
        #     print( "Removing a layer, now: %d layers" % layer_count )

        result = fixed_depth_exploration( target, num_qubits, gate_size,
                                          gate_fun_vals, gate_loc_vals,
                                          exploration_distance, learning_rate )

        success, gate_fun_vals, gate_loc_vals = result

        if success:

            # if verbosity >= 1:
            #     print( "Found a circuit with %d layers" % layer_count )

            found_gate_fun_vals = gate_fun_vals
            found_gate_loc_vals  = gate_loc_vals


    return found_gate_fun_vals, found_gate_loc_vals


def refinement ( target, num_qubits, gate_size, gate_fun_vals, gate_locs_fixed,
                 refinement_distance = 1e-7, learning_rate = 1e-6 ):
    """
    Refines circuit's distance with FixedGates instead of GenericGates.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        gate_fun_vals (List[List[float]]): Gate function values

        gate_locs_fixed (List[Tuple[int]]): Gate locations

        refinement_distance (float): Refinement's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (List[List[float]]): Refined gate function values
    """

    tf.reset_default_graph()

    layers = [ FixedGate( "Gate%d" % i, num_qubits, gate_size,
                          location = l, fun_values = a )
               for a, l in zip( gate_fun_vals, gate_locs_fixed ) ]

    tensor = layers[0].get_tensor()
    for layer in layers[1:]:
        tensor = tf.matmul( layer.get_tensor(), tensor )

    loss_fn   = hilbert_schmidt_distance( target, tensor )
    optimizer = tf.train.AdamOptimizer( learning_rate )
    train_op  = optimizer.minimize( loss_fn )
    init_op   = tf.global_variables_initializer()

    loss_values = []

    with tf.Session() as sess:
        sess.run( init_op )

        for i in range( 10000 ):
            for j in range( 50 ):
                loss = sess.run( [ train_op, loss_fn ] )[1]

            if loss < refinement_distance:
                break

            loss_values.append( loss )

            # if verbosity >= 2:
            #     print( loss )

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

        # if verbosity >= 1:
        #     print( "Refined circuit to %f error" % loss )

        return [ l.get_fun_vals( sess ) for l in layers ]
