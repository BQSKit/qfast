import tensorflow as tf
import numpy      as np
import itertools  as it

from qfast import LocationModel


class TestLocationModelConstructor ( tf.test.TestCase ):

    def test_locationmodel_invalid ( self ):
        self.assertRaises( ValueError, LocationModel, 0, 2 )
        self.assertRaises( ValueError, LocationModel, 4, 0 )
        self.assertRaises( ValueError, LocationModel, 4, 5 )
        self.assertRaises( ValueError, LocationModel, 0, 0 )
        self.assertRaises( ValueError, LocationModel, 4, 2, [(1, 2), (1, 2)] )
        self.assertRaises( ValueError, LocationModel, 3, 2, [(0, 1), (1, -1)] )
        self.assertRaises( ValueError, LocationModel, 3, 2, [(0, 1), (1, 3)] )
        self.assertRaises( ValueError, LocationModel, 3, 2, [(0, 1), (1, 1)] )
        self.assertRaises( ValueError, LocationModel, 3, 2, [(0, 1), (1, 2, 3)] )

    def test_locationmodel_invalid_bucketing_alg ( self ):

        def invalid_alg_1 ( locations ):
            return -1

        def invalid_alg_2 ( locations ):
            return []

        def invalid_alg_3 ( locations ):
            return [ locations, locations ]

        self.assertRaises( TypeError, LocationModel, 4, 2, None, invalid_alg_1 )
        self.assertRaises( ValueError, LocationModel, 4, 2, None, invalid_alg_2 )
        self.assertRaises( ValueError, LocationModel, 4, 2, None, invalid_alg_3 )

    def test_locationmodel_valid ( self ):
        lm = LocationModel( 4, 2 )
        self.assertEqual( lm.num_qubits, 4 )
        self.assertEqual( lm.gate_size, 2 )
        self.assertEqual( lm.locations, set( it.combinations( range( 4 ), 2 ) ) )
        self.assertEqual( lm.locations, set( lm.buckets[0] + lm.buckets[1] ) )

        lm = LocationModel( 6, 3 )
        self.assertEqual( lm.num_qubits, 6 )
        self.assertEqual( lm.gate_size, 3 )
        self.assertEqual( lm.locations, set( it.combinations( range( 6 ), 3 ) ) )
        self.assertEqual( lm.locations, set( lm.buckets[0] + lm.buckets[1] ) )

        def valid_alg ( locations ):
            locations = list( locations )
            n = len( locations )
            return [ locations[:n//2], locations[n//2:] ]

        lm = LocationModel( 4, 2, None, valid_alg )
        self.assertEqual( lm.num_qubits, 4 )
        self.assertEqual( lm.gate_size, 2 )
        self.assertEqual( lm.locations, set( it.combinations( range( 4 ), 2 ) ) )
        self.assertEqual( lm.locations, set( lm.buckets[0] + lm.buckets[1] ) )
        locations = list( lm.locations )
        n = len( locations )
        self.assertEqual( lm.buckets[0], locations[:n//2] )
        self.assertEqual( lm.buckets[1], locations[n//2:] )


if __name__ == '__main__':
    tf.test.main()
