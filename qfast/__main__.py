import os
import sys
import logging
import numpy    as np
import argparse as ap
import pickle

from qfast import *


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

    parser.add_argument( "-v", "--verbose",
                         default = 0,
                         action = "count",
                         help = "Verbose output" )

    parser.add_argument( "-m", "--model",
                         type = str,
                         default = "softpauli",
                         help = "The model to use." )

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

    handler.setLevel( logging.DEBUG )
    logger.addHandler( handler )

    if args.decompose_only:
        target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
        decomposer = Decomposer( target, model = args.model )
        gate_list = decomposer.decompose()
        # TODO Add parameters
        # TODO Change Pickle to unitary dump
        pickle.dump( gate_list, args.unitary_dir )

    elif args.instantiate_only:
        gate_list = pickle.load( args.unitary_dir )
        instantiater = Instantiater( args.native_tool )
        qasm_list = instantiater.instantiate( gate_list )
        # TODO Add parameters
        # TODO Change Pickle to unitary dump
        pickle.dump( qasm_list, args.qasm_dir )

    elif args.recombine_only:
        qasm_list = pickle.load( args.qasm_dir )
        combiner = Combiner()
        # TODO Same as above
        qasm_out = combiner.combine( qasm_list )

        with open( args.qasm_file, "w" ) as f:
            f.write( qasm_out )

    elif complete_qfast:
        target = np.loadtxt( args.unitary_file, dtype = np.complex128 )
        decomposer = Decomposer( target, model = args.model )
        gate_list = decomposer.decompose()
        instantiater = Instantiater( args.native_tool )
        qasm_list = instantiater.instantiate( gate_list )
        combiner = Combiner()
        qasm_out = combiner.combine( qasm_list )

        with open( args.qasm_file, "w" ) as f:
            f.write( qasm_out )

