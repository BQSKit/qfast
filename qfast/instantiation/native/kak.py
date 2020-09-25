"""
This module implements qiskit's kak as a native tool plugin to QFAST.
"""

import qiskit
import numpy as np

from qfast import utils
from qfast.instantiation import nativetool


class KAKTool ( nativetool.NativeTool ):
    """Synthesize tool built on QISKit's KAK implementation."""

    def get_maximum_size ( self ):
        """
        The maximum size of a unitary matrix (in qubits) that can be
        decomposed with this tool.

        Returns:
            (int): The qubit count this tool can handle.
        """

        return 2

    def synthesize ( self, utry ):
        """
        Synthesis function with QISKit's KAK implementation.

        Args:
            utry (np.ndarray): The unitary to synthesize.

        Returns
            qasm (str): The synthesized QASM output.

        Raises:
            TypeError: If utry is not a valid unitary.

            ValueError: If the utry has invalid dimensions.
        """

        if not utils.is_unitary( utry, tol = 1e-14 ):
            raise TypeError( "utry must be a valid unitary." )

        if utry.shape[0] > 2 ** self.get_maximum_size():
            raise ValueError( "utry has incorrect dimensions." )

        if utry.shape[0] == 4:
            circ = qiskit.QuantumCircuit( 2 )
            circ.unitary( utry, [ 1, 0 ] )
        else:
            circ = qiskit.QuantumCircuit( 1 )
            circ.unitary( utry )

        circ = qiskit.compiler.transpile( circ, basis_gates = ['u3', 'cx'],
                                          optimization_level = 3 )
        return circ.qasm()

