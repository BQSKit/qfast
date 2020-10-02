"""
This module implements the Topology class.

A Topology determines how the qubits are connected.
"""

import numpy     as np
import itertools as it

from qfast import utils

class Topology:
    """The Topology Class."""

    def __init__ ( self, num_qubits, coupling_graph = None ):
        """
        Constructs an all-to-all topology by default.

        Args:
            num_qubits (int): The total number of qubits in the topology.

            coupling_graph (List[Tuple[int]]): List of connected qubit pairs.

        Raises:
            TypeError: If coupling_graph is invalid.
        """

        if coupling_graph is None:
            coupling_graph = []
            for l in it.combinations( range( num_qubits ), 2 ):
                coupling_graph.append( tuple( l ) )

        if not utils.is_valid_coupling_graph( coupling_graph, num_qubits ):
            raise TypeError( "Invalid coupling graph." )

        self.coupling_graph = coupling_graph
        self.num_qubits = num_qubits
        self.cache = {}

        self.adjlist = [ [] for i in range( self.num_qubits ) ]
        for q0, q1 in coupling_graph:
            self.adjlist[q0].append( q1 )
            self.adjlist[q1].append( q0 )

    def get_locations ( self, gate_size ):
        """
        Returns a list of locations that complies with the topology.

        Each location has gate_size number of qubits. A location is only
        included if each pair of qubits is directly connected or connected
        through other qubits in the location.

        Args
            gate_size (int): The size of each location in the final list.

        Returns:
            (List[Tuple[int]]): The locations compliant with the topology.

        Raises:
            ValueError: If the gate_size is nonpositive or too large.
        """

        if gate_size > self.num_qubits:
            raise ValueError( "The gate_size is too large." )

        if gate_size <= 0:
            raise ValueError( "The gate_size is nonpositive." )

        if gate_size in self.cache:
            return self.cache[ gate_size ]
        
        locations = []

        for group in it.combinations( range( self.num_qubits ), gate_size ):
            # Depth First Search
            seen = set( [ group[0] ] )
            frontier = [ group[0] ]

            while len( frontier ) > 0 and len( seen ) < len( group ):
                for q in group:
                    if frontier[0] in self.adjlist[q] and q not in seen:
                        seen.add( q )
                        frontier.append( q )

                frontier = frontier[1:]

            if len( seen ) == len( group ):
                locations.append( group )

        self.cache[ gate_size ] = locations
        return locations

