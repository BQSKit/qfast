"""
This module implements the functions for the instantiation phase.
"""

import importlib
import pkgutil


class Instantiater():

    def __init__ ( self, tool ):
        if "qfast.native." + tool not in _discovered_tools:
            raise ValueError( "The native tool specified was not found." )

        self.tool = _discovered_tools[ "qfast.native." + tool ]

    def instantiate ( self, gate_list ):
        qasm_list = []

        for gate in gate_list:
            qasm = self.tool.synthesize( gate.utry )
            qasm_list.append( ( qasm, gate.get_location() ) )

        return qasm_list

