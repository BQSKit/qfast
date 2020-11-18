"""This module implements a naive combiner for QFAST."""

import re

from qfast.recombination import combiner


class NaiveCombiner ( combiner.Combiner ):
    """Naive Combiner performs no optimization."""

    def combine ( self, qasm_list ):
        """
        Combines the circuits into one circuit.

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

        max_qubit = 0
        combined_list = []

        for qasm, location in qasm_list:
            
            # Track max qubit index
            for qubit in location:
                if qubit > max_qubit:
                    max_qubit = qubit

            gate_list = self.parse_qasm( qasm )
            gate_list = self.apply_location( gate_list, location )
            combined_list += gate_list

        # Build final qasm string
        qasm  = "OPENQASM 2.0;\n"
        qasm += "include \"qelib1.inc\";\n"
        qasm += "qreg q[%d];\n" % (max_qubit + 1)
        qasm += self.to_qasm( combined_list )
        return qasm

    def parse_qasm ( self, qasm ):
        """
        Parses the simple qasm string

        Args:
            qasm (str): Input qasm string.

        Returns:
            (list[tuple[str, Tuple[int]]]): The list of gates and the
                qubits they act on.
        """

        reg_re = r'qreg\s*(?P<name>[a-zA-Z]+)\[(?P<qubits>\d+)\];'
        reg_ptrn = re.compile( reg_re )
        reg_name = None
        num_qubits = None

        gate_re = r'(?P<gate>[a-zA-Z0-9]+(\s*\([\w\s,\.\+\*\(\)\\\/\-\^]+\))?)'
        qubit_re = "%s\[(?P<idx>\d+)\]"
        gate_ptrn = re.compile( gate_re )
        qubit_ptrn = None

        gate_list = []

        for line in qasm.splitlines():

            # Skip header, empty lines, and commentted lines
            if "OPENQASM" in line:
                continue
            elif "include" in line:
                continue
            elif line.strip() == "":
                continue
            elif line[:2] == r"\\":
                continue

            # Parse qubit register declarations
            elif "qreg" in line:

                if reg_name is not None:
                    raise RuntimeError( "Cannot handle multiple"
                                        " register declarations." )

                # Match regular expression
                match = reg_ptrn.match( line )
                if match is None:
                    raise RuntimeError( "Failed to parse line: %s" % line )

                # Parse match
                reg_name = match.group( "name" )
                num_qubits = int( match.group( "qubits" ) )
                qubit_re = qubit_re % reg_name
                qubit_ptrn = re.compile( qubit_re )

            # Parse gate
            else:
                if qubit_ptrn is None:
                    raise RuntimeError( "No qubit register defined yet." )

                # Match regular expression
                gate_match = gate_ptrn.match( line )
                if gate_match is None:
                    raise RuntimeError( "Failed to parse line: %s" % line )

                # Parse match
                gate_name = gate_match.group( "gate" )
                qubits = qubit_ptrn.findall( line )
                qubits = tuple( [ int( qubit ) for qubit in qubits ] )
                gate_list.append( ( gate_name, qubits ) )

        return gate_list

    def apply_location ( self, gate_list, location ):
        """
        Transforms the gate_list by applying the specified location.

        Args:
            gate_list (list[tuple[str, Tuple[int]]): The list of gates.

            location (tuple[int]): The layout of the gates.

        Returns:
            (list[tuple[str, Tuple[int]]]): The gates with their qubits
                changed to match the location.
        """

        new_gate_list = []
        for gate_name, qubits in gate_list:
            new_qubits = tuple( [ location[ qubit ] for qubit in qubits ] )
            new_gate_list.append( ( gate_name, new_qubits ) )
        return new_gate_list

    def to_qasm ( self, gate_list ):
        """Convert a gate list to a qasm string."""
        qasm = ""

        for gate_name, qubits in gate_list:
            qasm += gate_name + " ";
            for qubit in qubits[:-1]:
                qasm += "q[%d]," % qubit
            qasm += "q[%d];\n" % qubits[-1]

        return qasm

