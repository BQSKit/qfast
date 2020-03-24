import tensorflow as tf
import numpy      as np
import itertools  as it

from qfast import LocationModel


class TestLocationModelConstructor ( tf.test.TestCase ):

    def test_locationmodel_fix_locations ( self ):
        lm = LocationModel( 4, 2 )
        loc_vals = [ [0, 1, 2, 3], [0, 1] ]
        loc_fixed = lm.fix_locations( loc_vals )
        self.assertEqual( loc_fixed[0], lm.buckets[1][3] )
        self.assertEqual( loc_fixed[1], lm.buckets[0][1] )

        lm = LocationModel( 4, 2 )
        loc_vals = [ [0, 1, 4, 3], [1, 0] ]
        loc_fixed = lm.fix_locations( loc_vals )
        self.assertEqual( loc_fixed[0], lm.buckets[1][2] )
        self.assertEqual( loc_fixed[1], lm.buckets[0][0] )


if __name__ == '__main__':
    tf.test.main()
