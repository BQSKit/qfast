"""
This module implements the functions for the instantiation phase.
"""

import importlib
import pkgutil


import qfast.native as native


discovered_tools = {
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

    return [ tool.split('.')[-1] for tool in discovered_tools.keys() ]


def get_native_tool ( tool ):
    """
    Retrieves the native tool's module from the discovered list.

    Args:
        tool (str): The native tool

    Return:
        (module): Module associated with the tool specified
    """

    if "qfast.native." + tool not in discovered_tools:
        raise ValueError( "The native tool specified was not found." )

    return discovered_tools[ "qfast.native." + tool ]


def instantiation ( tool, utry ):
    """
    Instantiation uses a native tool to convert a unitary into qasm.

    Args:
        tool (str): The native tool to use

        utry (np.ndarray): The unitary to synthesize

    Return:
        (str): qasm code that implements the unitary
    """

    m = get_native_tool( tool )

    if ( not hasattr( m, "get_native_block_size" ) or
         not hasattr( m, "synthesize" ) ):
        raise TypeError( "The native tool specified has an invalid or "
                         "incomplete api." )

    return m.synthesize( utry )

