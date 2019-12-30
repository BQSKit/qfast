import numpy           as np
import argparse        as ap
import scipy.linalg    as la
import search_compiler as sc

from timeit import default_timer as timer

from qfast import synthesize, refine_circuit, hilbert_schmidt_distance
from qfast import get_norder_paulis, pauli_dot_product, get_pauli_n_qubit_projection


def load_unitary ( unitary_name ):
    return np.loadtxt( unitary_name, dtype = np.complex128 )


def get_projection_unitary ( gate_params ):
    num_qubits = int( np.log2( len( gate_params ) ) / 2 )
    sigma = get_norder_paulis( num_qubits )
    alpha = gate_params
    H = pauli_dot_product( alpha, sigma )
    return la.expm( 1j * H )


def verify_circuit ( target, circuit ):
    num_qubits  = int( np.log2( len( target ) ) )

    acm = np.identity( 2 ** num_qubits )
    for link, alpha in circuit:
        sigma = get_pauli_n_qubit_projection( num_qubits, link )
        H = pauli_dot_product( alpha, sigma )
        assert( np.allclose( H, H.conj().T ) )
        acm = la.expm( 1j * H ) @ acm
        assert( np.allclose( acm @ acm.conj().T, np.identity(2 ** num_qubits) ) )
        assert( np.allclose( acm.conj().T @ acm, np.identity(2 ** num_qubits) ) )

    assert( hilbert_schmidt_distance( target, acm ) <= 0.03 )


def search_compile ( circuit ):
    compile_project = sc.Project( "__SC_Project__" )

    for i, gate in enumerate( circuit ):
        link, params = gate
        gate_target  = get_projection_unitary( params )
        name = str( i ) + "_" + str( link )
        compile_project.add_compilation( name, gate_target )

    compile_project.run()

    for i, gate in enumerate( circuit ):
        link, params = gate
        name = str( i ) + "_" + str( link )
        compile_project.assemble( name )


def qiskit_kak_compile ( ):
    pass


if __name__ == "__main__":
    description_info = "Synthesize a unitary matrix."

    parser = ap.ArgumentParser( description = description_info )

    parser.add_argument( "unitary_file", type = str,
                         help = "Unitary file to synthesize" )

    parser.add_argument( "circ_file", type = str,
                         help = "Circ output file" )

    # parser.add_argument( "kernel", type = str, default = "qiskit_KAK",
    #                      choices = [ "search_compiler", "qiskit_KAK" ] )

    parser.add_argument( "-t", "--test", action = "store_true",
                         help = "Verify circuit" )

    parser.add_argument( "-i", "--start_layer", type = int, default = 1,
                         help = "Number of layers to start with" )

    parser.add_argument( "-s", "--layer_step", type = int, default = 1,
                         help = "Number of layers to increase by every step" )

    parser.add_argument( "-r", "--refine", action = "store_true",
                         help = "Refine circuit" )

    # parser.add_argument( "-o", "--optimize", action = "store_true",
    #                      help = "Optimize circuit with qiskit" )

    parser.add_argument( "-b", "--block_size", type = int, default = 2,
                         help = "Synthesize into blocks of this size" )

    parser.add_argument( "-v", "--verbose", action = "count", default = 0,
                         help = "Verbose printouts" )

    args = parser.parse_args()

    target = load_unitary( args.unitary_file )
    target = target.astype( 'complex64' )

    num_qubits = int( np.log2( len( target ) ) )

    start = timer()
    circuit = synthesize( target, args.start_layer, args.layer_step,
                          args.block_size, args.verbose )
    end = timer()

    if args.verbose >= 1:
        print( "Found circuit in %f seconds" % end - start )

    if args.refine:
        target = target.astype( 'complex128' )
        start = timer()
        circuit = refine_circuit( target, circuit, args.verbose )
        end = timer()
        if args.verbose >= 1:
            print( "refined circuit in %f seconds" % end - start )

    with open( args.circ_file, "w" ) as f:
        f.write( str( circuit ) )

    # if num_qubits >= 8:
    #     gate_size = 5 if args.kernel == "search_compiler" else 4

    # elif num_qubits in [6, 7]:
    #     gate_size = 4

    # elif num_qubits == 5:
    #     gate_size = 3

    # elif num_qubits == 4:
    #     gate_size = 3 if args.kernel == "search_compiler" else 2

    # elif num_qubits == 3:
    #     if args.kernel == "search_compiler":
    #         target = target.astype( 'complex128' )
    #         search_compile( [(0,1,2), target] )
    #     gate_size = 2

    # elif num_qubits == 2:
    #     qiskit_kak_compile(  )

    # elif num_qubits <= 1:
    #     raise RuntimeError( "Only synthesizes circuits with >= 2 qubits." )

    # target = target.astype( 'complex64' )
    # start = timer()
    # circuit = synthesize( target, args.start_layer, args.layer_step, gate_size, args.verbose )
    # end = timer()
    # if args.verbose >= 1:
    #     print( "Found circuit in %f seconds", end - start )

    # if args.refine:
    #     target = target.astype( 'complex128' )
    #     start = timer()
    #     circuit = refine_circuit( target, circuit, args.verbose )
    #     end = timer()
    #     if args.verbose >= 1:
    #         print( "refined circuit in %f seconds", end - start )

    # if args.test:
    #     verify_circuit( target, circuit )

    # while ( gate_size > 3 or ( args.kernel == "qiskit_KAK" and gate_size > 2 ) ):

    #     if gate_size == 5:
    #         gate_size = 3

    #     elif gate_size == 4:
    #         gate_size = 3 if args.kernel == "search_compiler" else 2

    #     elif gate_size == 3:
    #         gate_size = 2

    #     if args.verbosity >= 1:
    #         print( "Compiling inner gates to size %d now." % gate_size )

    #     new_circuit = []

    #     for link, gate in circuit:
    #         gate_target  = get_projection_unitary( gate )
    #         gate_target = gate_target.astype( 'complex64' )
    #         gate_circuit = synthesize( gate_target, 1, args.layer_step, gate_size, args.verbose )

    #         if args.refine:
    #             gate_target = gate_target.astype( 'complex128' )
    #             gate_circuit = refine_circuit( gate_target, gate_circuit, args.verbose )

    #         if args.test:
    #             verify_circuit( gate_target, gate_circuit )

    #         for link2, gate2 in gate_circuit:
    #             new_link = [ link[l] for l in link2 ]
    #             new_gate = gate2
    #             new_circuit.append( ( new_link, new_gate ) )

    #     circuit = new_circuit

    #     if args.test:
    #         verify_circuit( target, circuit )

    # if args.kernel == "search_compiler":
    #     search_compile( circuit )

    # else:
    #     qiskit_kak_compile( ... )
