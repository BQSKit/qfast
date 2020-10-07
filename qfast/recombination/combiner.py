"""
This module implements the Combiner class.

The combiner puts together small circuits into a big one.
"""

import logging

from qiskit import *


logger = logging.getLogger( "qfast" )

class Combiner():

    def __init__ ( self, optimization = True ):
        """
        Initializes a combiner

        Args:
            optimization (bool): Whether or not optimization is on.
        """

        self.optimization = optimization

        logger.debug( "Starting Recombination." )

    def combine ( self, qasm_list ):
        """
        Combiners the circuits into one circuit.

        Args:
            qasm_list(List[Tuple[str, Tuple[int]]]): The small circuits
                and their locations.

        Returns:
            (str): The final circuit's QASM
        """

        if not isinstance( qasm_list, list ):
            raise TypeError( "qasm_list must be a list." )

        if not all( isinstance( qasm[0], str ) for qasm in qasm_list ):
            raise TypeError( "qasm_list must contain QASM strings." )

        loc_fixed = [ qasm[1] for qasm in qasm_list ]
        qasm_list = [ qasm[0] for qasm in qasm_list ]

        # Calculate Output Circuit Size
        max_qubit = 0

        for loc in loc_fixed:
            for qub in loc:
                max_qubit = max( qub, max_qubit )

        # Convert to circuits
        circs = []
        for qasm in qasm_list:
            try:
                circs.append( QuantumCircuit.from_qasm_str( qasm ) )
            except:
                raise ValueError( "Invalid QASM string: %s" % qasm )

        if all( len( circ.qubits ) != len( loc )
                for circ, loc in zip( circs, loc_fixed ) ):
            raise ValueError( "Location and QASM qubit counts don't match." )

        # Join into One Circuit
        out_circ = QuantumCircuit( max_qubit + 1 )

        for circ, loc in zip( circs, loc_fixed ):
            for gate in circ.data:
                if gate[0].name == 'cx':
                    out_circ.cx( loc[ gate[1][0].index ], loc[ gate[1][1].index ] )
                elif gate[0].name == 'u1':
                    out_circ.u1( *gate[0].params, loc[ gate[1][0].index ] )
                elif gate[0].name == 'u2':
                    out_circ.u2( *gate[0].params, loc[ gate[1][0].index ] )
                elif gate[0].name == 'u3':
                    out_circ.u3( *gate[0].params, loc[ gate[1][0].index ] )
                elif gate[0].name == 'rx':
                    out_circ.rx( *gate[0].params, loc[ gate[1][0].index ] )
                elif gate[0].name == 'ry':
                    out_circ.ry( *gate[0].params, loc[ gate[1][0].index ] )
                elif gate[0].name == 'rz':
                    out_circ.rz( *gate[0].params, loc[ gate[1][0].index ] )
                else:
                    raise ValueError( "QASM must be in \'u1, u2, u3, cx\' basis." )

        out_circ = qiskit.compiler.transpile( out_circ, basis_gates = ['u3', 'cx'],
                                              optimization_level = 3 if self.optimization else 0 )
        return out_circ.qasm()

