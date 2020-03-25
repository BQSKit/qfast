import tensorflow as tf
import numpy      as np
import itertools  as it

from qfast import LocationModel


class TestLocationModelConstructor ( tf.test.TestCase ):

    def test_locationmodel_fix_locations ( self ):
        lm = LocationModel( 4, 2 )
        loc_vals = [ [0, 1, 2], [0, 1, 2] ]
        loc_fixed = lm.fix_locations( loc_vals )
        self.assertEqual( loc_fixed[0], lm.buckets[1][2] )
        self.assertEqual( loc_fixed[1], lm.buckets[0][2] )

        lm = LocationModel( 4, 2 )
        loc_vals = [ [0, 3, 2], [0, 1, 2] ]
        loc_fixed = lm.fix_locations( loc_vals )
        self.assertEqual( loc_fixed[0], lm.buckets[1][1] )
        self.assertEqual( loc_fixed[1], lm.buckets[0][2] )


if __name__ == '__main__':
    tf.test.main()
