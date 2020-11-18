__version__ = "2.1.0"

# Initialize Logging
import logging
_logger = logging.getLogger( "qfast" )
_logger.setLevel(logging.CRITICAL)
_handler = logging.StreamHandler()
_handler.setLevel( logging.DEBUG )
#_fmt = "%(levelname)-8s | %(message)s (%(module)s:%(funcName)s:%(lineno)s)"
_fmt = "%(levelname)-8s | %(message)s"
_formatter = logging.Formatter( _fmt )
_handler.setFormatter( _formatter )
_logger.addHandler( _handler )

# Initialize Plugins
modelsubclasses = {}
optimizersubclasses = {}
nativetoolsubclasses = {}
combinersubclasses = {}

# Main API
from .decomposition.decomposer import Decomposer
from .instantiation.instantiater import Instantiater
from .recombination.combiner import Combiner
from .synthesis import synthesize
