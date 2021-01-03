import numpy    as np
import unittest as ut

from qfast.topology import Topology
from qfast.instantiation.instantiater import Instantiater

class TestInstantiaterConstructor ( ut.TestCase ):

    def test_instantiater_constructor_invalid ( self ):
        invalid_tool = "test_dummy_tool"
        valid_topology = Topology( 3, None )

        self.assertRaises( RuntimeError, Instantiater, invalid_tool, valid_topology )

    def test_instantiater_constructor_invalid2 ( self ):
        valid_tool = "QSearchTool"
        invalid_topology = "a"

        self.assertRaises( TypeError, Instantiater, valid_tool, invalid_topology )

    def test_instantiater_constructor_valid ( self ):
        valid_tool = "QSearchTool"
        valid_topology = Topology( 3, None )
        instantiater = Instantiater( valid_tool, valid_topology )
        self.assertTrue( instantiater.tool.__class__.__name__ == valid_tool )


if __name__ == '__main__':
    ut.main()

