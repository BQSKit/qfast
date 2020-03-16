"""
This module implements the main decomposition functions.
"""

import tensorflow as tf
import numpy      as np
import itertools  as it

from decomposition import FixedGate, GenericGate, Block
from tools import hilbert_schmidt_distance, get_unitary_from_pauli_coefs


def decomposition ( block, **kwargs ):
    """
    Decomposes a Block into a sequence of smaller blocks which
    implement the input.

    Args:
        block (Block): Target Block

    Keyword Args:
        start_depth (int): The number of gates to start exploring from

        depth_step (int): The number of added gates in each
                          exploration step

        exploration_distance (float): Exploration's goal distance

        exploration_learning_rate (float): Learning rate of
                                           exploration's optimizer

        refinement_distance (float): Refinement's goal distance

        refinement_learning_rate (float): Learning rate of refinement's
                                          optimizer

    Returns:
        (List[Block]): Decomposed blocks
    """

    if block.num_qubits <= 2:
        return [ block ]

    params = {}
    params["start_depth" ] = 1
    params["depth_step" ] = 1
    params["exploration_distance" ] = 0.01
    params["exploration_learning_rate" ] = 0.01
    params["refinement_distance" ] = 1e-7
    params["refinement_learning_rate" ] = 1e-6
    params.update( kwargs )

    gate_size = get_decomposition_size( block.num_qubits )

    fun_vals, loc_vals = exploration( block.utry, block.num_qubits, gate_size,
                                      params["start_depth"],
                                      params["depth_step"],
                                      params["exploration_distance"],
                                      params["exploration_learning_rate"] )

    loc_fixed = fix_locations( block.num_qubits, gate_size, loc_vals )

    fun_vals = refinement( block.utry, block.num_qubits, gate_size,
                           fun_vals, loc_fixed,
                           params["refinement_distance"],
                           params["refinement_learning_rate"] )

    return convert_to_block_list( block.get_location(), fun_vals, loc_fixed )


def get_decomposition_size ( num_qubits ):
    """
    Given a block size, computes the block size to decompose to

    Args:
        num_qubits (int): Number of qubits in block

    Returns:
        (int): Decomposed block size
    """

    if not isinstance( num_qubits, int ):
        raise TypeError( "Input must be an integer." )

    if num_qubits <= 0:
        raise ValueError( "Number of qubits must be positive." )

    return int( np.ceil( num_qubits / 2 ) )


def exploration ( target, num_qubits, gate_size, start_depth, depth_step,
                  exploration_distance = 0.01, learning_rate = 0.01 ):
    """
    Synthesizes a circuit that implements the target unitary. Explores
    both circuit structure (gate location) and gate functions.

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
            The final fun_vals and loc_vals
    """

    depth = start_depth
    fun_vals = [ None ] * depth
    loc_vals = [ None ] * depth

    # print( "Starting to search with %d layers" % layer_count )

    # Stride search over layer_count
    while ( True ):
        result = fixed_depth_exploration( target, num_qubits, gate_size,
                                          fun_vals, loc_vals,
                                          exploration_distance, learning_rate )

        success, fun_vals, loc_vals = result

        if success:

            # if verbosity >= 1:
            #     print( "Found a circuit with %d layers" % layer_count )

            # circuit = result[1]
            break

        fun_vals += [ None ] * depth_step
        loc_vals += [ None ] * depth_step
        depth += depth_step

        # if verbosity >= 1:
        #     print( "Added a layer, now: %d layers" % layer_count )

    # Remember good results
    found_fun_vals = fun_vals
    found_loc_vals = loc_vals

    # Search backwards for a shorter circuit
    for i in range( depth_step - 1 ):
        depth -= 1

        # if verbosity >= 1:
        #     print( "Removing a layer, now: %d layers" % layer_count )

        result = fixed_depth_exploration( target, num_qubits, gate_size,
                                          fun_vals, loc_vals,
                                          exploration_distance, learning_rate )

        success, fun_vals, loc_vals = result

        if success:

            # if verbosity >= 1:
            #     print( "Found a circuit with %d layers" % layer_count )

            found_fun_vals = fun_vals
            found_loc_vals = loc_vals

    return found_fun_vals, found_loc_vals


def fixed_depth_exploration ( target, num_qubits, gate_size, fun_vals,
                              loc_vals, exploration_distance = 0.01,
                              learning_rate = 0.01 ):
    """
    Attempts to synthesize the target unitary with a fixed number
    of gates of size gate_size.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        fun_vals (List[List[float]]): Gate function values

        loc_vals (List[List[float]]): Gate location values

        exploration_distance (float): Exploration's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (Tuple[bool, List[List[float]], List[List[float]]]):
            True if succeeded in hitting exploration distance
            and the final fun_vals and loc_vals
    """

    tf.reset_default_graph()

    layers = [ GenericGate( "Gate%d" % i, num_qubits, gate_size,
                            fun_vals[i], loc_vals[i],
                            parity = (i + 1) % 2 )
               for i in range( len( fun_vals ) ) ]

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


def fix_locations ( num_qubits, gate_size, loc_vals ):
    """
    Fixes the locations; used when converting from generic gates to
    fixed gates.

    Args:
        num_qubits (int): Total number of qubits

        gate_size (int): The size of the gates

        loc_vals (List[List[float]]): Gate unfixed location values

    Returns:
        (List[Tuple[int]]): Fixed locations

    Note:
        This is bad programing since it's duplicated code
        from different parts of the program and is more or less a hack
        TODO: write a HardwareModel class that models a target
        hardware architecture and can be queried for this and
        other information.
    """

    loc_fixed = []
    topology = list( it.combinations( range( num_qubits ), gate_size ) )

    for i, loc_val in enumerate( loc_vals ):
        loc_idx = np.argmax( loc_val )
        parity = (i + 1) % 2
        if parity == 0:
            loc_fixed.append( topology[:len(topology)//2][loc_idx] )
        elif parity == 1:
            loc_fixed.append( topology[len(topology)//2:][loc_idx] )

    return loc_fixed


def refinement ( target, num_qubits, gate_size, fun_vals, loc_fixed,
                 refinement_distance = 1e-7, learning_rate = 1e-6 ):
    """
    Refines synthesized circuit to better implement the target unitary.
    This is achieved by using fixed circuit structure (gate location)
    and a more fine-grained optimizer.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        fun_vals (List[List[float]]): Gate function values

        loc_fixed (List[Tuple[int]]): Gate locations

        refinement_distance (float): Refinement's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (List[List[float]]): Refined gate function values
    """

    tf.reset_default_graph()

    layers = [ FixedGate( "Gate%d" % i, num_qubits, gate_size,
                          loc = loc_fixed[i], fun_vals = fun_vals[i] )
               for i in range( len( fun_vals ) ) ]

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


def convert_to_block_list ( block_loc, fun_vals, loc_fixed ):
    """
    Converts the function parameters to unitary matrices and composes
    location into a larger circuit. Returns the resulting block list.

    Args:
        block_loc (Tuple[int]): The location in a larger circuit that
                                this circuit corresponds to.

        fun_vals (List[List[float]]): Gate function values

        locs_fixed (List[Tuple[int]]): Gate locations

    Returns:
        (List[Block]): Resulting block list
    """

    block_list = []

    for loc, fun_params in zip( loc_fixed, fun_vals ):

        # Convert to unitary
        utry = get_unitary_from_pauli_coefs( fun_params )

        # Compose location
        new_loc = tuple( [ block_loc[i] for i in loc ] )

        block_list.append( Block( utry, new_loc ) )

    return block_list
