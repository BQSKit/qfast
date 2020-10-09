"""
This module defines the NativeTool abstract base class.

A native tool converts gates in a generic gate set into native gates.

All native tool plugins must extend the NativeTool class and implement the
functionality outlined here.
"""


import abc
import qfast


class NativeToolMeta ( abc.ABCMeta ):
    """The NativeTool Metaclass."""

    def __init__ ( cls, name, bases, attr ):
        """Automatically registers native tool plugins with qfast."""

        qfast.nativetoolsubclasses[name] = cls
        super().__init__( name, bases, attr )


class NativeTool ( metaclass = NativeToolMeta ):
    """The NativeTool abstract base class."""

    @abc.abstractmethod
    def get_maximum_size ( self ):
        """
        Returns the maximum size this native tool can handle.

        Returns:
            (int): The largest number of qubits this tool can handle.
        """
        pass

    @abc.abstractmethod
    def synthesize ( self, utry ):
        """
        Synthesize the unitary input into native gates.

        Args:
            utry (np.ndarray): The unitary to synthesize.

        Returns:
            (str): The synthesized QASM output.
        """
        pass

