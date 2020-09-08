import numpy     as np
import itertools as it

class Topology:

    def __init__ ( self, num_qubits, coupling_map = None ):
        self.coupling_map = coupling_map
        self.num_qubits = num_qubits
        
        if self.coupling_map is None:
            self.coupling_map = [ tuple(l) for l in it.combinations( range( self.num_qubits ), 2 ) ]

        self.topology_cache = {}

    def get_locations ( self, target_gate_size ):
        if target_gate_size in self.topology_cache:
            return self.topology_cache[ target_gate_size ]
        
        topology = []

        for group in it.combinations( range( self.num_qubits ), target_gate_size ):
            # Depth First Search
            seen = set( [ group[0] ] )
            frontier = [ group[0] ]

            while len( frontier ) > 0 and len( seen ) < len( group ):
                # TODO Prepare graph in adjacency list format
                for q in group:
                    if ( frontier[0], q ) in self.coupling_map or ( q, frontier[0] ) in self.coupling_map:
                        if q not in seen:
                            seen.add( q )
                            frontier.append( q )

                frontier = frontier[1:]

            if len( seen ) == len( group ):
                topology.append( group )

        self.topology_cache[ target_gate_size ] = topology
        return topology

