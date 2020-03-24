import tensorflow as tf
import numpy      as np
import itertools  as it

from qfast import greedy_max_cut


class TestGreedyMaxCut ( tf.test.TestCase ):

    def check_greedy_max_cut_props ( self, num_qubits, gate_size ):
        locs = set( it.combinations( range( num_qubits ), gate_size ) )
        buckets = greedy_max_cut( locs )
        self.assertEqual( len( buckets ), 2 )
        self.assertEqual( len( set( buckets[0] ).intersection( set( buckets[1] ) ) ), 0 )
        self.assertEqual( len( set( buckets[0] ).union( set( buckets[1] ) ) ), len( locs ) )
        self.assertTrue( all( [ loc in locs for loc in buckets[0] ] ) )
        self.assertTrue( all( [ loc in locs for loc in buckets[1] ] ) )

    def test_greedy_max_cut ( self ):
        test_cases = [ (  4, 2 ), (  5, 2 ), (  6, 2 ), (  7, 2 ), (  8, 2 ),
                       (  9, 2 ), ( 10, 2 ), ( 11, 2 ), ( 12, 2 ), ( 13, 2 ),
                       (  4, 2 ), (  5, 3 ), (  6, 3 ), (  7, 3 ), (  8, 3 ),
                       (  9, 2 ), ( 10, 3 ), ( 11, 3 ), ( 12, 3 ), ( 13, 3 ),
                       (  4, 4 ), (  5, 4 ), (  6, 4 ), (  7, 4 ), (  8, 4 ),
                       (  9, 4 ), ( 10, 4 ), ( 11, 4 ), ( 12, 4 ), ( 13, 4 ) ]

        for num_qubits, gate_size in test_cases:
            self.check_greedy_max_cut_props( num_qubits, gate_size )


if __name__ == '__main__':
    tf.test.main()
