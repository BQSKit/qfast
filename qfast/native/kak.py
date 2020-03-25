"""
This module implements qiskit's kak as a native tool plugin to QFAST.
"""

import qiskit
import numpy as np


def get_native_block_size():
    """
    The maximum size of a unitary matrix (in qubits) that can be
    decomposed with this module.

    Returns:
        (int): The qubit count this module can handle.
    """

    return 2


def synthesize ( utry ):
    """
    Synthesis function with QISKit's KAK implementation.

    Args:
        utry (np.ndarray): The unitary matrix to synthesize.

    Returns
        qasm (str): The synthesized QASM output.
    """

    if not isinstance( utry, np.ndarray ):
        raise TypeError( "utry must be a np.ndarray." )

    if len( utry.shape ) != 2:
        raise TypeError( "utry must be a matrix." )

    if utry.shape[0] != 2 ** get_native_block_size():
        raise ValueError( "utry has incorrect dimensions." )

    if utry.shape[1] != 2 ** get_native_block_size():
        raise ValueError( "utry has incorrect dimensions." )

    if ( not np.allclose( utry.conj().T @ utry, np.identity( len( utry ) ),
                          rtol = 0, atol = 1e-14 )
         or
         not np.allclose( utry @ utry.conj().T, np.identity( len( utry ) ),
                          rtol = 0, atol = 1e-14 ) ):
        raise ValueError( "utry must be a unitary matrix." )

    circ = qiskit.QuantumCircuit( get_native_block_size() )
    circ.unitary( utry, [ 1, 0 ] )
    circ = qiskit.compiler.transpile( circ, basis_gates = ['u3', 'cx'],
                                      optimization_level = 3 )
    return circ.qasm()


