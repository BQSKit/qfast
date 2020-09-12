"""
This module defines the CircuitModel abstract base class.

A CircuitModel structures a large unitary as products of smaller
unitaries. This enables the decomposition of unitaries. 

All model plugins must extend this class and implement the
functionality outlined here.
"""

import abc

import numpy as np

import qfast
from qfast import utils


class ModelMeta ( abc.ABCMeta ):
    """The CircuitModel Metaclass."""

    def __init__ ( cls, name, bases, attr ):
        """Automatically registers model plugins with qfast."""

        qfast.modelsubclasses[name] = cls
        super().__init__( name, bases, attr )


class CircuitModel ( metaclass = ModelMeta ):
    """The CircuitModel abstract base class."""
    
    def __init__ ( self, utry, gate_size, locations, optimizer ):
        """
        Default constructor for CircuitModels.

        Args:
            utry (np.ndarray): The unitary to decompose.

            gate_size (int): The size of the smaller unitaries.

            locations (List[Tuple[int]]): The valid locations for gates.

            optimizer (Optimizer): The optimizer available for use.

        Raises:
            ValueError: If the gate_size or locations are invalid.
        """

        self.num_qubits = int( np.log2( len( utry ) ) )

        if gate_size >= self.num_qubits:
            raise ValueError( "Invalid gate_size" )

        self.gate_size = gate_size

        if not utils.is_valid_locations( locations, num_qubits, gate_size ):
            raise TypeError( "Invalid locations" )

        self.locations = locations
        self.optimizer = optimizer
        self.utry = utry
        self.utry_dag = utry.conj().T
    
    @abc.abstractmethod
    def solve ( self ):
        """
        Solve the decomposition problem defined in this CircuitModel.

        Returns:
            (List[Gate]): The list of gates that implement the unitary.
        """
        pass

