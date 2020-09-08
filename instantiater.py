"""
This module implements the functions for the instantiation phase.
"""

import importlib
import pkgutil


import qfast.native as native


_discovered_tools = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( native.__path__, native.__name__ + "." )
}


def list_native_tools():
    """
    List the discovered native tools.

    Returns
        (List[str]): List of discovered tools
    """

    return [ tool.split('.')[-1] for tool in _discovered_tools.keys() ]


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

