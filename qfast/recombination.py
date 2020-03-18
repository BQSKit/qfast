"""
This module implements the functions for the recombination phase.
"""

from qiskit import *

def recombination ( qasm_list, loc_fixed ):
    """
    Recombines the circuits in qasm_list into one circuit.

    Args:
        qasm_list (List[str]): The list of circuits given by QASM

        loc_fixed (List[Tuple[int]]): The circuit locations

    Returns:
        (str): The final circuit's QASM
    """
    
    # Calculate Output Circuit Size
    max_qubit = 0

    for loc in loc_fixed:
        for qub in loc:
            max_qubit = max( qub, max_qubit )
    
    # Convert locations for qiskit ordering
    loc_corrected = []
    for loc in loc_fixed:
        loc_corrected.append( [ max_qubit - value for value in loc ] )
    
    # Convert to circuits
    circs = [ QuantumCircuit.from_qasm_str( qasm ) for qasm in qasm_list ]

    # Join into One Circuit
    out_circ = QuantumCircuit( max_qubit + 1 )

    for circ, loc in zip( circs, loc_corrected ):
        for gate in circ.data:
            if gate[0].name == 'cx':
                out_circ.cx( loc[ gate[1][0].index ], loc[ gate[1][1].index ] )
            elif gate[0].name == 'u1':
                out_circ.u1( *gate[0].params, loc[ gate[1][0].index ] )
            elif gate[0].name == 'u2':
                out_circ.u2( *gate[0].params, loc[ gate[1][0].index ] )
            elif gate[0].name == 'u3':
                out_circ.u3( *gate[0].params, loc[ gate[1][0].index ] )

    out_circ = qiskit.compiler.transpile( out_circ, basis_gates = ['u3', 'cx'],
                                          optimization_level = 3 )
    return out_circ.qasm()

