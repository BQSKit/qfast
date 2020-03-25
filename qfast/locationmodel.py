"""
This module implements the location model class.

The location model is responsible for mapping location indices
to locations. This is useful in gate parity calculations.
"""

import numpy as np
import itertools as it


def greedy_max_cut ( locations ):
    """
    Partitions the locations into two buckets by placing them as nodes
    in a graph, where edges connect two locations with at least one
    shared qubit, and greedily approximating a maximum cut.

    Args:
        locations (Set[Tuple[int]]): The location set

    Returns:
        buckets (List[List[Tuple[int]]]): The bucketed locations
    """

    # Create Graph as adjacency list, store cut info in nodes
    graph = { l:[] for l in locations }
    nodes = { l:{ "cut":0, "uncut":0 } for l in locations }
    edges = set()
    for n1, n2 in it.product( locations, repeat = 2 ):
        if n1 == n2:
            continue
        for q in n1:
            if q in n2 and ( n2, n1 ) not in edges:
                edges.add( ( n1, n2 ) )
                graph[ n1 ].append( n2 )
                graph[ n2 ].append( n1 )
                break

    # Start with empty cut
    for node, adj_node_list in graph.items():
        nodes[ node ][ "uncut" ] = len( adj_node_list )

    buckets = [ [], list( nodes.keys() ) ]

    while True:
        # Sort nodes by possible value gain
        sorted_nodes = sorted( nodes.items(),
                               key = lambda x: x[1]["uncut"] - x[1]["cut"],
                               reverse = True )

        nd = sorted_nodes[0]

        # If no more possible value gain, stop
        if nd[1]["uncut"] - nd[1]["cut"] <= 0:
            break

        # Swap nd across the cut
        buckets[0].append( nd[0] )
        buckets[1].remove( nd[0] )
        temp = nodes[ nd[0] ][ 'cut' ]
        nodes[ nd[0] ][ 'cut' ] = nodes[ nd[0] ][ 'uncut' ]
        nodes[ nd[0] ][ 'uncut' ] = temp
        for neighbor in graph[ nd[0] ]:
            nodes[ neighbor ][ 'cut' ] += 1
            nodes[ neighbor ][ 'uncut' ] -= 1

    return buckets


def lexicographical_cut ( locations ):
    """
    Partitions the locations into two buckets by cutting in half.

    Args:
        locations (Set[Tuple[int]]): The location set

    Returns:
        buckets (List[List[Tuple[int]]]): The bucketed locations
    """

    locations = list( locations )
    n = len( locations )
    return [ locations[:n//2], locations[n//2:] ]


class LocationModel():
    """The LocationModel class."""

    def __init__ ( self, num_qubits, gate_size, cgraph = None,
                   bucketing_alg = None ):
        """
        Location model construction.

        Args:
            num_qubits (int): Number of qubits in the target unitary

            gate_size (int): Number of qubits in each gate

            cgraph (Set[Tuple[int]]): Coupling graph, defaults to
                                      fully connected hardware topology

            bucketing_alg (Set[Tuple[int]] -> List[List[Tuple[int]]]):
                Function that converts a set of locations into buckets
                of locations. Defaults to lexicographical cut.
        """

        if num_qubits <= 0:
            raise ValueError( "Invalid qubit count: %d." % num_qubits )

        if gate_size > num_qubits or gate_size <= 0:
            raise ValueError( "Invalid gate size: %d." % gate_size )

        if cgraph is None:
            cgraph = set( it.combinations( range( num_qubits ), 2 ) )

        if len( cgraph ) != len( set( cgraph ) ):
            raise ValueError( "cgraph contains duplicates: %s", str( cgraph ) )

        if ( not all( [ ( 0 <= q < num_qubits ) for l in cgraph for q in l ] )
             or not all( [ len( l ) == len( set( l ) ) for l in cgraph ] )
             or not all( [ len( l ) == 2 for l in cgraph ] ) ):
            raise ValueError( "Invalid cgraph supplied: %s.", str( cgraph ) )

        if bucketing_alg is None:
            bucketing_alg = lexicographical_cut

        self.num_qubits = num_qubits
        self.gate_size = gate_size
        self.cgraph = cgraph
        self.bucketing_alg = bucketing_alg

        # TODO Topology Aware: change locations to be a function of cgraph
        self.locations = it.combinations( range( num_qubits ), gate_size )
        self.locations = set( self.locations )

        self.buckets = bucketing_alg( self.locations )
        self.num_buckets = len( self.buckets )

        # Check if bucketing algorithm returned valid buckets.
        bucket_sum = set()
        element_sum = 0
        for b in self.buckets:
            bucket_sum.update( b )
            element_sum += len( b )
        if bucket_sum != self.locations or element_sum != len( self.locations ):
            raise ValueError( "Invalid bucketing algorithm, not a partition." )

    def fix_locations ( self, loc_vals ):
        """
        Fixes the locations; used when converting from generic gates to
        fixed gates.

        Args:
            loc_vals (List[List[float]]): Gate unfixed location values

        Returns:
            (List[Tuple[int]]): Fixed locations
        """

        loc_fixed = []

        for i, loc_val in enumerate( loc_vals ):
            loc_idx = np.argmax( loc_val )
            parity = (i + 1) % self.num_buckets
            loc_fixed.append( self.buckets[ parity ][ loc_idx ] )

        return loc_fixed
