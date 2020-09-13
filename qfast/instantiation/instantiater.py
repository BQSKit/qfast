"""
This module implements the Instantiater class.

The instantiater uses a native tool to convert small generic gates
to a native gate set.
"""


from qfast import plugins


class Instantiater():
    """The Instantiater Class."""

    def __init__ ( self, tool ):
        """
        Construct an instantiater with a native tool.

        Args:
            tool (str): The name of the native tool to use.

        Raises:
            RuntimeError: If the native tool cannot be found.
        """

        if tool not in plugins.get_native_tools():
            raise RuntimeError( f"Cannot find native tool: {tool}" )

        self.tool = plugins.get_native_tool( tool )()

    def instantiate ( self, gate_list ):
        """
        Perform the instantiation phase.

        Args:
            gate_list (list[Gates]): The list of generic gates.

        Returns:
            (list[tuple[str, tuple[int]]]): List of qasm and
                gate locations.
        """

        qasm_list = []

        for gate in gate_list:
            qasm = self.tool.synthesize( gate.utry )
            qasm_list.append( ( qasm, gate.location ) )

        return qasm_list

