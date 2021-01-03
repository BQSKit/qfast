"""
This module implements the Instantiater class.

The instantiater uses a native tool to convert small generic gates
to a native gate set.
"""

import logging

from qfast import gate
from qfast import plugins
from qfast.topology import Topology


logger = logging.getLogger( "qfast" )


class Instantiater():
    """The Instantiater Class."""

    def __init__ ( self, tool, topology, basis_gates = None ):
        """
        Construct an instantiater with a native tool.

        Args:
            tool (str): The name of the native tool to use.

            topology (Topology): The topology of the circuit.

            basis_gates (List[str]): The two-qubit gate native gate.

        Raises:
            RuntimeError: If the native tool cannot be found.
        """

        if tool not in plugins.get_native_tools():
            raise RuntimeError( f"Cannot find native tool: {tool}" )

        if not isinstance( topology, Topology ):
            raise TypeError( "Invalid topology" )

        self.tool = plugins.get_native_tool( tool )()
        self.topology = topology
        self.basis_gates = basis_gates

    def instantiate ( self, gate_list ):
        """
        Perform the instantiation phase.

        Args:
            gate_list (list[Gates]): The list of generic gates.

        Returns:
            (list[tuple[str, tuple[int]]]): List of qasm and
                gate locations.
        """

        if not isinstance( gate_list, list ):
            raise TypeError( "Invalid gate list." )

        if not all( [ isinstance( g, gate.Gate ) for g in gate_list ] ):
            raise TypeError( "Invalid gate list." )

        logger.debug( "Starting Instantiation with %s."
                      % self.tool.__class__.__name__ )

        qasm_list = []

        for g in gate_list: 
            coupling_graph = self.topology.get_subgraph( g.location )
            renum_map = { q:i for i, q in enumerate(g.location) }
            coupling_graph = [ (renum_map[i], renum_map[j]) for i, j in coupling_graph ]
            qasm = self.tool.synthesize( g.utry,
                                         basis_gates = self.basis_gates, 
                                         coupling_graph = coupling_graph )
            qasm_list.append( ( qasm, g.location ) )

        return qasm_list

