import numpy    as np
import unittest as ut

from qfast.instantiation.instantiater import Instantiater

class TestInstantiaterConstructor ( ut.TestCase ):

    def test_instantiater_constructor_invalid ( self ):
        invalid_tool = "test_dummy_tool"

        self.assertRaises( RuntimeError, Instantiater, invalid_tool )

    def test_instantiater_constructor_valid ( self ):
        valid_tool = "KAKTool"
        instantiater = Instantiater( valid_tool )
        self.assertTrue( instantiater.tool.__class__.__name__ == valid_tool )


if __name__ == '__main__':
    ut.main()

