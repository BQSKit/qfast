"""Plugin module is a central point of access for all loaded plugins."""

import importlib
import pkgutil

import qfast
from qfast.decomposition import models
from qfast.decomposition import optimizers
from qfast.instantiation import native
from qfast.recombination import combiners


_discovered_models = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( models.__path__,
                             models.__name__ + "." )
}

_discovered_optimizers = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( optimizers.__path__,
                             optimizers.__name__ + "." )
}

_discovered_native_tools = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( native.__path__,
                             native.__name__ + "." )
}

_discovered_combiners = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( combiners.__path__,
                             combiners.__name__ + "." )
}



def get_models():
    """
    List the discovered circuit models.

    Returns:
        (List[str]): List of models
    """

    models = list( qfast.modelsubclasses.keys() )
    models.remove( 'CircuitModel' )
    return models


def get_optimizers():
    """
    List the discovered optimizers.

    Returns:
        (List[str]): List of optimizers
    """

    optimizers = list( qfast.optimizersubclasses.keys() )
    optimizers.remove( 'Optimizer' )
    return optimizers


def get_native_tools():
    """
    List the discovered native tools.

    Returns:
        (List[str]): List of discovered native tools
    """

    nativetools = list( qfast.nativetoolsubclasses.keys() )
    nativetools.remove( 'NativeTool' )
    return nativetools


def get_combiners():
    """
    List the discovered combiners.

    Returns:
        (List[str]): List of discovered combiners
    """

    combiners = list( qfast.combinersubclasses.keys() )
    combiners.remove( 'Combiner' )
    return combiners


def get_model ( name ):
    """
    Retrieves a circuit model.

    Args:
        name (str): The retrieved model's name.

    Returns
        (CircuitModel): The retrieved model.
    """

    return qfast.modelsubclasses[ name ]


def get_optimizer ( name ):
    """
    Retrieves an optimizer.

    Args:
        name (str): The retrieved optimizer's name.

    Returns
        (Optimizer): The retrieved optimizer.
    """

    return qfast.optimizersubclasses[ name ]


def get_native_tool ( name ):
    """
    Retrieves a native tool.

    Args:
        name (str): The retrieved native tool's name.

    Returns
        (NativeTool): The retrieved native tool.
    """

    return qfast.nativetoolsubclasses[ name ]


def get_combiner ( name ):
    """
    Retrieves a combiner.

    Args:
        name (str): The retrieved combiner's name.

    Returns
        (Combiner): The retrieved combiner.
    """

    return qfast.combinersubclasses[ name ]

