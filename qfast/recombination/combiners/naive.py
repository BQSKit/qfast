"""This module implements a naive combiner for QFAST."""

import re

from qfast import utils
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

        # 1. For each qasm in qasm_list, apply location
        # 2. Put qasm together

        max_qubit = 0
        combined_list = []

        for qasm, location in qasm_list:

            for qubit in location:
                if qubit > max_qubit:
                    max_qubit = qubit

            gate_list = self.parse_qasm( qasm )
            gate_list = self.apply_location( gate_list, location )
            combined_list += gate_list

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

        reg_regex = r'qreg\s*(?P<name>[a-zA-Z]+)\[(?P<qubits>\d+)\];'
        reg_pattern = re.compile( reg_regex )
        reg_name = None
        num_qubits = None

        gate_regex = r'(?P<gate>[a-zA-Z0-9]+(\s*\([\w\s,\.\+\*\(\)\\\/\-\^]+\))?)'
        qubit_regex = "%s\[(?P<idx>\d+)\]"
        gate_pattern = re.compile( gate_regex )
        qubit_pattern = None

        gate_list = []

        for line in qasm.splitlines():
            if "OPENQASM" in line:
                continue
            elif "include" in line:
                continue
            elif "qreg" in line:
                if reg_name is not None:
                    raise RuntimeError( "Cannot handle multiple register declarations." )
                match = reg_pattern.match( line )
                if match is None:
                    raise RuntimeError( "Failed to parse line: %s" % line )
                reg_name = match.group( "name" )
                num_qubits = int( match.group( "qubits" ) )
                qubit_regex = qubit_regex % reg_name
                qubit_pattern = re.compile( qubit_regex )
            elif line.strip() == "" or line[:2] == r"\\":
                continue
            else:
                if qubit_pattern is None:
                    raise RuntimeError( "No qubit register defined yet." )
                gate_match = gate_pattern.match( line )
                if gate_match is None:
                    raise RuntimeError( "Failed to parse line: %s" % line )
                gate_name = gate_match.group( "gate" )
                qubits = qubit_pattern.findall( line )
                qubits = tuple( [ int( qubit ) for qubit in qubits ] )
                gate_list.append( ( gate_name, qubits ) )

        return gate_list

    def apply_location ( self, gate_list, location ):
        new_gate_list = []
        for gate_name, qubits in gate_list:
            new_qubits = tuple( [ location[ qubit ] for qubit in qubits ] )
            new_gate_list.append( ( gate_name, new_qubits ) )
        return new_gate_list

    def to_qasm ( self, gate_list ):
        qasm = ""

        for gate_name, qubits in gate_list:
            qasm += gate_name + " ";
            for qubit in qubits[:-1]:
                qasm += "q[%d]," % qubit
            qasm += "q[%d];\n" % qubits[-1]

        return qasm

