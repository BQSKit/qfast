from abc import ABC, abstractmethod

import numpy as np

class CircuitModel ( ABC ):
    
    def __init__ ( self, utry, gate_size, locations, optimizer ):
        self.utry = utry
        self.utry_dag = utry.conj().T
        self.num_qubits = int( np.log2( len( utry ) ) )
        self.gate_size = gate_size
        self.locations = locations
        self.optimizer = optimizer
    
    @abstractmethod
    def solve ( self ):
        pass

