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

    circ = qiskit.QuantumCircuit( get_native_block_size() )
    circ.unitary( utry, [ 1, 0 ] )
    circ = qiskit.compiler.transpile( circ, basis_gates = ['u3', 'cx'],
                                      optimization_level = 3 )
    return circ.qasm()


