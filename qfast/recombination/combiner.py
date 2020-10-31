"""
This module defines the Combiner abstract base class.

A combiner puts circuits together.

All combiner plugins must extend the Combiner class and implement the
functionality outlined here.
"""


import abc
import qfast


class CombinerMeta ( abc.ABCMeta ):
    """The Combiner Metaclass."""

    def __init__ ( cls, name, bases, attr ):
        """Automatically registers combiner plugins with qfast."""

        qfast.combinersubclasses[name] = cls
        super().__init__( name, bases, attr )


class Combiner ( metaclass = CombinerMeta ):
    """The Combiner abstract base class."""

    @abc.abstractmethod
    def combine ( self, qasm_list ):
        """
        Combines the circuits into one circuit.

        Args:
            qasm_list(List[Tuple[str, Tuple[int]]]): The small circuits
                and their locations.

        Returns:
            (str): The final circuit's QASM
        """
        pass

