import os
import sys
import logging
import numpy    as np
import argparse as ap

from .circuit import Circuit
from .instantiation import *
from .recombination import recombination


logger = logging.getLogger( "qfast" )


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

    parser.add_argument( "-n", "--native-tool",
                         type = str,
                         default = None,
                         choices = list_native_tools(),
                         help = "The tool to use during instantiation." )

    parser.add_argument( "-b", "--block-size",
                         type = int,
                         default = 2,
                         help = "The block size to decompose to." )

    parser.add_argument( "-s", "--start-depth", dest = "sd",
                         type = int,
                         default = 1,
                         help = "The initial depth to start exploring from." )

    parser.add_argument( "-d", "--depth-step", dest = "ds",
                         type = int,
                         default = 1,
                         help = ( "The number of gates to append every"
                                  "step of exploration." ) )

    parser.add_argument( "-e", "--exploration-distance", dest = "ed",
                         type = float,
                         default = 0.01,
                         help = "The distance tolerance for exploration." )

    parser.add_argument( "-l", "--exploration-learning-rate", dest = "el",
                         type = float,
                         default = 0.01,
                         help = "The learning rate for exploration." )

    parser.add_argument( "-r", "--refinement-distance", dest = "rd",
                         type = float,
                         default = 1e-7,
                         help = "The distance tolerance for refinement." )

    parser.add_argument( "-j", "--refinement-learning-rate", dest = "rl",
                         type = float,
                         default = 1e-6,
                         help = "The learning rate for refinement." )

    parser.add_argument( "-v", "--verbose",
                         default = 0,
                         action = "count",
                         help = "Verbose output" )

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
            parser.error( "No unitary input file specified"
                          ", add --unitary-file." )
        if args.qasm_file is None:
            parser.error( "No qasm output file specified"
                          ", add --qasm-file." )
        if args.native_tool is None:
            parser.error( "No native tool specified"
                          ", add --native-tool." )

    # If we are only doing decomposition need unitary file and dir
    if args.decompose_only:
        if args.unitary_file is None:
            parser.error( "No unitary input file specified"
                          ", add --unitary-file." )
        if args.unitary_dir is None:
            parser.error( "No unitary output directory specified"
                          ", add --unitary-dir." )

    # If we are only doing instantiation need unitary dir and qasm dir
    if args.instantiate_only:
        if args.unitary_dir is None:
            parser.error( "No unitary input directory specified"
                          ", add --unitary-dir." )
        if args.qasm_dir is None:
            parser.error( "No qasm output directory specified"
                          ", add --qasm-dir." )
        if args.native_tool is None:
            parser.error( "No native tool specified"
                          ", add --native-tool." )

    # If we are only doing recombination need qasm dir and qasm file
    if args.recombine_only:
        if args.qasm_dir is None:
            parser.error( "No qasm input directory specified"
                          ", add --qasm-dir." )
        if args.qasm_file is None:
            parser.error( "No qasm output file specified"
                          ", add --qasm-file." )

    # Logging Init
    handler = logging.StreamHandler( sys.stdout )

    if args.verbose >= 3:
        logger.setLevel( logging.DEBUG )
    elif args.verbose == 2:
        handler.addFilter( lambda m: m.getMessage()[:4] != "Loss" )
        logger.setLevel( logging.DEBUG )
    elif args.verbose == 1:
        logger.setLevel( logging.INFO )
    elif args.verbose == 0:
        logger.setLevel( logging.CRITICAL )
        logging.getLogger( "tensorflow" ).addFilter( lambda m: False )

    handler.setLevel( logging.DEBUG )
    logger.addHandler( handler )

    if args.decompose_only:
        target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
        circ = Circuit( target )
        circ.hierarchically_decompose( args.block_size,
                                       start_depth = args.sd,
                                       depth_step = args.ds,
                                       exploration_distance = args.ed,
                                       exploration_learning_rate = args.el,
                                       refinement_distance = args.rd,
                                       refinement_learning_rate = args.rl )

        if not os.path.isdir( args.unitary_dir ):
            os.makedirs( args.unitary_dir )

        circ.dump_blocks( args.unitary_dir )

    elif args.instantiate_only:
        if not os.path.isdir( args.unitary_dir ):
            raise RuntimeError( "Unitary directory does not exist." )

        if not os.path.isdir( args.qasm_dir ):
            os.makedirs( args.qasm_dir )

        for file in os.listdir( args.unitary_dir ):
            utry = np.loadtxt( os.path.join( args.unitary_dir, file ),
                               dtype = np.complex128 )
            name = ".".join( file.split(".")[:-1] ) + ".qasm"
            qasm = instantiation( args.native_tool, utry )
            with open( os.path.join( args.qasm_dir, name ), 'w' ) as f:
                f.write( qasm )

    elif args.recombine_only:
        if not os.path.isdir( args.qasm_dir ):
            raise RuntimeError( "Qasm directory does not exist." )

        qasm_list_dict = {}
        loc_fixed_dict = {}

        for file in os.listdir( args.qasm_dir ):
            with open( os.path.join( args.qasm_dir, file ), "r" ) as f:
                qasm = f.read()
            name_list = file.replace( ".qasm", "" ).split("_")
            loc = tuple( int(x) for x in name_list[1:] )
            idx = int( name_list[0] )
            qasm_list_dict[ idx ] = qasm
            loc_fixed_dict[ idx ] = loc

        qasm_list = []
        loc_fixed = []

        for i in range( len( os.listdir( args.qasm_dir ) ) ):
            qasm_list.append( qasm_list_dict[ i ] )
            loc_fixed.append( loc_fixed_dict[ i ] )

        out_qasm = recombination( qasm_list, loc_fixed )

        with open( args.qasm_file, "w" ) as f:
            f.write( out_qasm )

    elif complete_qfast:
        target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
        circ = Circuit( target )
        block_size = get_native_tool( args.native_tool ).get_native_block_size()
        circ.hierarchically_decompose( block_size,
                                       start_depth = args.sd,
                                       depth_step = args.ds,
                                       exploration_distance = args.ed,
                                       exploration_learning_rate = args.el,
                                       refinement_distance = args.rd,
                                       refinement_learning_rate = args.rl )

        qasm_list = [ instantiation( args.native_tool, block.utry )
                      for block in circ.blocks ]
        locations = circ.get_locations()

        out_qasm = recombination( qasm_list, locations )

        with open( args.qasm_file, "w" ) as f:
            f.write( out_qasm )
