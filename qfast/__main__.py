"""QFAST Command Line Interface Module"""

import logging
import numpy as np
import argparse as ap

from qfast import plugins
from qfast import synthesize


logger = logging.getLogger( "qfast" )


if __name__ == "__main__":
    description_info = "Synthesize a unitary matrix."

    parser = ap.ArgumentParser( description = description_info )

    parser.add_argument( "unitary",
                         type = str,
                         help = "The unitary input file" )

    parser.add_argument( "qasm",
                         type = str,
                         help = "The qasm output file" )

    parser.add_argument( "-n", "--native-tool",
                         type = str,
                         default = "KAKTool",
                         choices = plugins.get_native_tools(),
                         help = "The tool to use during instantiation." )

    parser.add_argument( "-v", "--verbose",
                         default = 0,
                         action = "count",
                         help = "Verbose output" )

    parser.add_argument( "-m", "--model",
                         type = str,
                         default = "PermModel",
                         choices = plugins.get_models(),
                         help = "The model to use during decomposition." )

    parser.add_argument( "-o", "--optimizer",
                         type = str,
                         default = "LFBGSOptimizer",
                         choices = plugins.get_optimizers(),
                         help = "The optimizer to use during decomposition." )

    args = parser.parse_args()

    # Logger init
    if args.verbose == 2:
        logger.setLevel( logging.DEBUG )
    elif args.verbose == 1:
        logger.setLevel( logging.INFO )

    # Load input unitary
    try:
        utry = np.loadtxt( args.unitary, dtype = np.complex128 )
    except Exception as ex:
        raise RuntimeError( "Cannot load unitary input." ) from ex

    qasm_out = synthesize( utry, model = args.model,
                           optimizer = args.optimizer,
                           tool = args.native_tool )

    with open( args.qasm, "w" ) as f:
        f.write( qasm_out )

