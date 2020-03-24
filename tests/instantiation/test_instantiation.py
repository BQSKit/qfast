import tensorflow as tf
import numpy      as np

from qfast import instantiation, list_native_tools


class TestInstantiation ( tf.test.TestCase ):

    CNOT = np.array( [ [ 1, 0, 0, 0 ],
                       [ 0, 1, 0, 0 ],
                       [ 0, 0, 0, 1 ],
                       [ 0, 0, 1, 0 ] ], dtype = np.complex128 )

    def test_instantiation_invalid ( self ):
        native_tools = list_native_tools()

        invalid_tool = "test_dummy_tool"

        while invalid_tool in native_tools:
            invalid_tool += "a"

        self.assertRaises( ValueError, instantiation, invalid_tool, self.CNOT )

    def test_instantiation_valid ( self ):

        qasm = instantiation( "kak", self.CNOT )
        self.assertTrue( isinstance( qasm, str ) )
        self.assertTrue( "OPENQASM" in qasm )
        self.assertTrue( "cx" in qasm )
        self.assertTrue( "qreg q[2]" in qasm)


if __name__ == '__main__':
    tf.test.main()