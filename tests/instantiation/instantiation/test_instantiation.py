import numpy    as np
import unittest as ut

from qfast.instantiation.instantiater import Instantiater


class TestInstantiaterInstantiate ( ut.TestCase ):

    CNOT = np.array( [ [ 1, 0, 0, 0 ],
                       [ 0, 1, 0, 0 ],
                       [ 0, 0, 0, 1 ],
                       [ 0, 0, 1, 0 ] ], dtype = np.complex128 )

    def test_instantiater_constructor_invalid ( self ):
        invalid_tool = "test_dummy_tool"

        self.assertRaises( RuntimeError, Instantiater, invalid_tool )

    def test_instantiater_constructor_valid ( self ):
        valid_tool = "KAKTool"
        instantiater = Instantiater( valid_tool )
        self.assertTrue( instantiater.tool == valid_tool )

    def test_instantiater_instantiate_invalid ( self ):
        valid_tool = "KAKTool"
        instantiater = Instantiater( valid_tool )

        test_0 = 0
        test_1 = "a"
        test_2 = ( 0, 1 )
        test_3 = ( CNOT, 0 )
        test_4 = ( CNOT, ( 0, 1 ) )
        test_5 = [ ( CNOT, ( 0, 1 ) ) ]
        
        self.assertRaises( TypeError, instantiater.instantiate, test_0 )
        self.assertRaises( TypeError, instantiater.instantiate, test_1 )
        self.assertRaises( TypeError, instantiater.instantiate, test_2 )
        self.assertRaises( TypeError, instantiater.instantiate, test_3 )
        self.assertRaises( TypeError, instantiater.instantiate, test_4 )
        self.assertRaises( TypeError, instantiater.instantiate, test_5 )
        
    def test_instantiater_instantiate_valid ( self ):
        valid_tool = "KAKTool"
        instantiater = Instantiater( valid_tool )

        qasm = instantiater.instantiate( self.CNOT )
        self.assertTrue( isinstance( qasm, str ) )
        self.assertTrue( "OPENQASM" in qasm )
        self.assertTrue( "cx" in qasm )
        self.assertTrue( "qreg q[2]" in qasm)


if __name__ == '__main__':
    ut.main()

