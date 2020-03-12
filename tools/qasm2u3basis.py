import argparse
import numpy as np
from qiskit import *


def test_circ_equivalences ( circ1, circ2 ):
    circ1 = circ1.copy()
    circ2 = circ2.copy()
    circ1.remove_final_measurements()
    circ2.remove_final_measurements()
    backend = BasicAer.get_backend( "unitary_simulator" )
    utry1 = qiskit.execute( circ1, backend ).result().get_unitary()
    utry2 = qiskit.execute( circ2, backend ).result().get_unitary()

    if not np.allclose( utry1, utry2 ):
        raise RuntimeError( "Transpiled circuit failed verification." )


if __name__ == "__main__":
    description_info = "Convert QASM to another QASM file containing" + \
                       " only u3 and cx gates."

    parser = argparse.ArgumentParser( description = description_info )

    parser.add_argument( "input_file", type = str,
                         help = "QASM input file" )

    parser.add_argument( "output_file", type = str,
                         help = "Qasm output file" )

    parser.add_argument( "-t", "--test", action = 'store_true',
                         help = "Test output" )

    parser.add_argument( "-m", "--measure", action = 'store_true',
                         help = "Add measurements to output circuit." )

    args = parser.parse_args()

    circ_in = QuantumCircuit.from_qasm_file( args.input_file )
    circ_in.remove_final_measurements()

    circ_out = qiskit.compiler.transpile( circ_in, basis_gates = ['u3', 'cx'] )

    if args.measure:
        circ_out.measure_all()

    if args.test:
        test_circ_equivalences( circ_in, circ_out )

    with open( args.output_file, 'w' ) as f:
        f.write( circ_out.qasm() )
