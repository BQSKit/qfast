import tensorflow as tf
import numpy      as np


from qfast import GenericGate, LocationModel


class TestGenericgateConstructor ( tf.test.TestCase ):

    def test_genericgate_constructor_invalid ( self ):
        lm = LocationModel( 4, 2 )
        self.assertRaises( ValueError, GenericGate, "Test", 2, 4, lm )
        self.assertRaises( ValueError, GenericGate, "Test", 4, 2, lm, [0, 1] )
        self.assertRaises( ValueError, GenericGate, "Test", 4, 2, lm, [0]*16, [0, 1] )

    def test_genericgate_constructor_valid ( self ):
        lm = LocationModel( 4, 2 )
        gg = GenericGate( "Test", 4, 2, lm )
        self.assertEqual( gg.name, "Test" )
        self.assertEqual( gg.num_qubits, 4 )
        self.assertEqual( gg.gate_size, 2 )
        self.assertTrue( np.array_equal( gg.loc_vals, [ 0 ] * 6 ) )
        self.assertTrue( np.array_equal( gg.fun_vals, [ 0.25 ] * 16 ) )
        self.assertTrue( np.array_equal( list( lm.locations ), gg.topology ) )

        gg = GenericGate( "Test", 4, 2, lm, parity = 0 )
        self.assertEqual( gg.name, "Test" )
        self.assertEqual( gg.num_qubits, 4 )
        self.assertEqual( gg.gate_size, 2 )
        self.assertTrue( np.array_equal( gg.fun_vals, [ 0.25 ] * 16 ) )
        self.assertTrue( np.array_equal( lm.buckets[0], gg.topology ) )


if __name__ == '__main__':
    tf.test.main()
