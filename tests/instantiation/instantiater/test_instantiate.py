import numpy    as np
import unittest as ut

from qfast import gate
from qfast.instantiation.instantiater import Instantiater


class TestInstantiaterInstantiate ( ut.TestCase ):

    CNOT = np.array( [ [ 1, 0, 0, 0 ],
                       [ 0, 1, 0, 0 ],
                       [ 0, 0, 0, 1 ],
                       [ 0, 0, 1, 0 ] ], dtype = np.complex128 )

    def test_instantiater_instantiate_invalid ( self ):
        valid_tool = "QSearchTool"
        instantiater = Instantiater( valid_tool )

        test_0 = 0
        test_1 = "a"
        test_2 = ( 0, 1 )
        test_3 = ( self.CNOT, 0 )
        test_4 = ( self.CNOT, ( 0, 1 ) )
        test_5 = [ ( self.CNOT, ( 0, 1 ) ) ]
        
        self.assertRaises( TypeError, instantiater.instantiate, test_0 )
        self.assertRaises( TypeError, instantiater.instantiate, test_1 )
        self.assertRaises( TypeError, instantiater.instantiate, test_2 )
        self.assertRaises( TypeError, instantiater.instantiate, test_3 )
        self.assertRaises( TypeError, instantiater.instantiate, test_4 )
        self.assertRaises( TypeError, instantiater.instantiate, test_5 )
        
    def test_instantiater_instantiate_valid ( self ):
        valid_tool = "QSearchTool"
        instantiater = Instantiater( valid_tool )

        qasm_list = instantiater.instantiate( [ gate.Gate( self.CNOT, (0, 1) ) ] )

        self.assertTrue( isinstance( qasm_list, list ) )
        self.assertTrue( len( qasm_list ) == 1 )

        self.assertTrue( isinstance( qasm_list[0], tuple ) )
        self.assertTrue( len( qasm_list[0] ) == 2 )

        self.assertTrue( isinstance( qasm_list[0][0], str ) )
        self.assertTrue( "OPENQASM" in qasm_list[0][0] )
        self.assertTrue( "cx" in qasm_list[0][0] )
        self.assertTrue( "qreg q[2]" in qasm_list[0][0] )

        self.assertTrue( qasm_list[0][1] == (0, 1) )


if __name__ == '__main__':
    ut.main()

