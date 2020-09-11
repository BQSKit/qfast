from .decomposer import Decomposer
from .instantiater import Instantiater
from .instantiater import list_native_tools
from .combiner import Combiner


# Initialize Logging
import logging
_logger = logging.getLogger( "qfast" )
_logger.setLevel(logging.CRITICAL)
_handler = logging.StreamHandler()
_handler.setLevel( logging.DEBUG )
_fmt = "%(levelname)-8s | %(message)s (%(funcName)s:%(lineno)s)"
_formatter = logging.Formatter( _fmt )
_handler.setFormatter( _formatter )
_logger.addHandler( _handler )

