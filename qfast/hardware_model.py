"""
This module implements the HardwareModel class.

The hardware model contains hardware-specific information, such as
size and topology. This information can be constructed to target a
specific machine. By default, it assumes a fully connected topology.
"""

import itertools as it


class HardwareModel():
    """The HardwareModel class."""

    def __init__ ( self, size, topology = None ):
        """
        HardwareModel Construction.

        Args:
            size (int): The number of qubits in the model.

            topology (List[Tuple[int]]): The coupling topology.
                Defaults to fully connected unidirectional topology.
        """

        if size <= 0:
            raise ValueError( "Positive size required." )

        if topology is None:
            topology = list( it.combinations( range( size ), 2 ) )

        for link in topology:
            for qubit in link:
                if qubit >= size:
                    raise ValueError( "Invalid qubit in topology." )

        self.size     = size
        self.topology = topology

    def get_link ( self, index ):
        """Maps index to links."""
        return self.topology[ index ]

    def get_index ( self, link ):
        """Maps links to indices."""
        return self.topology.index( link )

    def get_half_topology ( self, which_half ):
        """
        Splits the topology in half.

        Args:
            which_half (int): 0 for first half, 1 for second half

        Returns:
            (List[Tuple[int]]) List of links in the half asked for
        """

        if which_half == 0:
            return self.topology[:len(self.topology)//2]
        elif which_half == 1:
            return self.topology[len(self.topology)//2:]
