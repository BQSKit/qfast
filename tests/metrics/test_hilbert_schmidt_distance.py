import tensorflow as tf
import numpy      as np

from qfast import hilbert_schmidt_distance


class TestHilbertSchmidtDistance ( tf.test.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    INVALID = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_hilbert_schmidt_distance_invalid_type ( self ):
        self.assertRaises( TypeError, hilbert_schmidt_distance,
                           1, 1 )
        self.assertRaises( TypeError, hilbert_schmidt_distance,
                           self.TOFFOLI, 1 )
        self.assertRaises( TypeError, hilbert_schmidt_distance,
                           1, self.TOFFOLI )
        toffoli_tensor = tf.constant( self.TOFFOLI )
        self.assertRaises( TypeError, hilbert_schmidt_distance,
                           toffoli_tensor, 1 )
        self.assertRaises( TypeError, hilbert_schmidt_distance,
                           1, toffoli_tensor )

    def test_hilbert_schmidt_distance_invalid_shape ( self ):
        self.assertRaises( ValueError, hilbert_schmidt_distance,
                           self.TOFFOLI, self.INVALID )
        self.assertRaises( ValueError, hilbert_schmidt_distance,
                           self.INVALID, self.TOFFOLI )

    def test_hilbert_schmidt_distance_numpy_numpy ( self ):
        loss = hilbert_schmidt_distance( self.TOFFOLI, self.TOFFOLI )
        self.assertEquals( loss, 0 )

    def test_hilbert_schmidt_distance_numpy_tensor ( self ):
        toffoli_tensor = tf.constant( self.TOFFOLI )
        loss = hilbert_schmidt_distance( self.TOFFOLI, toffoli_tensor )

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run( init_op )
            self.assertEquals( loss.eval(), 0 )

    def test_hilbert_schmidt_distance_tensor_numpy ( self ):
        toffoli_tensor = tf.constant( self.TOFFOLI )
        loss = hilbert_schmidt_distance( toffoli_tensor, self.TOFFOLI )

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run( init_op )
            self.assertEquals( loss.eval(), 0 )

    def test_hilbert_schmidt_distance_tensor_tensor ( self ):
        toffoli_tensor = tf.constant( self.TOFFOLI )
        loss = hilbert_schmidt_distance( toffoli_tensor, toffoli_tensor )

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run( init_op )
            self.assertEquals( loss.eval(), 0 )


if __name__ == '__main__':
    tf.test.main()
