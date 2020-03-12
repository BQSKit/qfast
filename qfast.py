import os
import numpy    as np
import argparse as ap

from decomposition import Circuit


if __name__ == "__main__":
    description_info = "Synthesize a unitary matrix."

    parser = ap.ArgumentParser( description = description_info )

    parser.add_argument( "--decompose-only",
                         default = False,
                         action = "store_true",
                         help = "Perform decomposition" )

    parser.add_argument( "--instantiate-only",
                         default = False,
                         action = "store_true",
                         help = "Perform instantiation" )

    parser.add_argument( "--recombine-only",
                         default = False,
                         action = "store_true",
                         help = "Perform recombination" )

    parser.add_argument( "--unitary-file",
                         type = str,
                         default = None,
                         help = "Unitary file input" )

    parser.add_argument( "--unitary-dir",
                         type = str,
                         default = None,
                         help = "Directory of unitary files" )

    parser.add_argument( "--qasm-dir",
                         type = str,
                         default = None,
                         help = "Directory of qasm files" )

    parser.add_argument( "--qasm-file",
                         type = str,
                         default = None,
                         help = "Qasm file output" )

    parser.add_argument( "-k", "--kernel", type = str, default = "kak",
                         choices = [ "uq", "kak" ] )


    args = parser.parse_args()

    # True if we have to do everything
    complete_qfast = not ( args.decompose_only or
                           args.instantiate_only or
                           args.recombine_only )

    # Can only have one "only" flag set
    if int( args.decompose_only ) + \
       int( args.instantiate_only ) + \
       int( args.recombine_only ) > 1:
        parser.error( "Can only have one stage-only flag set" )

    # If we need to do everything need unitary file and qasm file
    if complete_qfast:
        if args.unitary_file is None:
            parser.error( "No unitary input file specified" + \
                          ", add --unitary-file" )
        if args.qasm_file is None:
            parser.error( "No qasm output file specified" + \
                          ", add --qasm-file" )
 
    # If we are only doing decomposition need unitary file and dir
    if args.decompose_only:
        if args.unitary_file is None:
            parser.error( "No unitary input file specified" + \
                          ", add --unitary-file" )
        if args.unitary_dir is None:
            parser.error( "No unitary output directory specified" + \
                          ", add --unitary-file" )

    # If we are only doing instantiation need unitary dir and qasm dir
    if args.instantiate_only:
        if args.unitary_dir is None:
            parser.error( "No unitary input directory specified" + \
                          ", add --unitary-dir" )
        if args.qasm_dir is None:
            parser.error( "No qasm output directory specified" + \
                          ", add --qasm-dir" )

    # If we are only doing recombination need qasm dir and qasm file
    if args.recombine_only:
        if args.qasm_dir is None:
            parser.error( "No qasm output directory specified" + \
                          ", add --qasm-dir" )
        if args.qasm_file is None:
            parser.error( "No qasm output file specified" + \
                          ", add --qasm-file" )

    if args.decompose_only:
        target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
        circ   = Circuit( target ).decompose( 1 )
        circ.dump_blocks( args.unitary_dir )
    elif args.instantiate_only:
        pass
    elif args.recombine_only:
        pass
    elif complete_qfast:
        target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
        circ   = Circuit( target ).synthesize( 1 )
        pass

"""
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
            """

