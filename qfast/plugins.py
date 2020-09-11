import importlib
import pkgutil

import qfast.native
import qfast.models
import qfast.optimizers


_discovered_models = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( qfast.models.__path__,
                             qfast.models.__name__ + "." )
}

_discovered_optimizers = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( qfast.optimizers.__path__,
                             qfast.optimizers.__name__ + "." )
}

_discovered_native_tools = {
    name: importlib.import_module( name )
    for finder, name, ispkg
    in pkgutil.iter_modules( qfast.native.__path__,
                             qfast.native.__name__ + "." )
}


