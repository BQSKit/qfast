from abc import ABC, abstractmethod

class Optimizer ( ABC ):
    
    @abstractmethod
    def minimize_coarse ( self, objective_fn, xin ):
        pass

    @abstractmethod
    def minimize_fine ( self, objective_fn, xin ):
        pass

