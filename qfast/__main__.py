import os
import numpy           as np
import argparse        as ap
import scipy.linalg    as la

from timeit import default_timer as timer

from qfast import Block, Circuit
from qfast import synthesize, refine_circuit, hilbert_schmidt_distance
from qfast import get_norder_paulis, pauli_dot_product, get_pauli_n_qubit_projection


if __name__ == "__main__":
    description_info = "Synthesize a unitary matrix."

    parser = ap.ArgumentParser( description = description_info )

    parser.add_argument( "unitary_file", type = str,
                         help = "Unitary file to synthesize" )

    parser.add_argument( "output", type = str,
                         help = "Output File/Directory" )

    parser.add_argument( "-k", "--kernel", type = str, default = "kak",
                         choices = [ "uq", "kak" ] )


    args = parser.parse_args()

    target = np.loadtxt( args.unitary_file, dtype = np.complex128 )

    result = Circuit( target, args.kernel ).synthesize( 1 )

    if args.kernel == "uq":
        if not os.path.isdir( args.output ):
            os.mkdir( args.output )

        for i, block in enumerate( result ):
            linkname = str( block.link ).replace( ", ", "_" ).replace("(", "").replace(")", "")
            filename = "%d_%s.unitary" % ( i, linkname )
            filename = os.path.join( args.output, filename )
            np.savetxt( filename, block.utry )
    else:
        with open( args.output, 'w' ) as f:
            f.write( result )
