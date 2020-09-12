import abc

import numpy as np
import qfast

class ModelMeta ( abc.ABCMeta ):

    def __init__ ( cls, name, bases, attr ):
        qfast.modelsubclasses[name] = cls
        super().__init__( name, bases, attr )

class CircuitModel ( metaclass = ModelMeta ):
    
    def __init__ ( self, utry, gate_size, locations, optimizer ):
        self.utry = utry
        self.utry_dag = utry.conj().T
        self.num_qubits = int( np.log2( len( utry ) ) )
        self.gate_size = gate_size
        self.locations = locations
        self.optimizer = optimizer
    
    @abc.abstractmethod
    def solve ( self ):
        pass

