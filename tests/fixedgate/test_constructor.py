import tensorflow as tf
import numpy      as np


from qfast import FixedGate


class TestFixedgateConstructor ( tf.test.TestCase ):

    def test_fixedgate_constructor_invalid ( self ):
        self.assertRaises( ValueError, FixedGate, "Test", 2, 4, (0, 1, 2, 3) )
        self.assertRaises( ValueError, FixedGate, "Test", 4, 2, (0, 1), [0, 1] )

    def test_fixedgate_constructor_valid ( self ):
        fg = FixedGate( "Test", 4, 2, (0, 1) )
        self.assertEqual( fg.name, "Test" )
        self.assertEqual( fg.num_qubits, 4 )
        self.assertEqual( fg.gate_size, 2 )
        self.assertTrue( np.array_equal( fg.location, (0, 1) ) )
        self.assertTrue( np.array_equal( fg.fun_vals, [ 0.25 ] * 16 ) )


if __name__ == '__main__':
    tf.test.main()
