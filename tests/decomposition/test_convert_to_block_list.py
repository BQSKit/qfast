import tensorflow as tf
import numpy      as np

from qfast import Block, Circuit, convert_to_block_list


class TestDecompositionConvertToBlockList ( tf.test.TestCase ):

    def test_decomposition_convert_to_block_list ( self ):
        block_loc = ( 1, 2, 3 )
        fun_vals = [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                     [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ]
        loc_fixed = [ (0, 1), (1, 2) ]
        block_list = convert_to_block_list( block_loc, fun_vals, loc_fixed )
        self.assertEqual( len( block_list ), 2 )
        self.assertEqual( block_list[0].loc, (1, 2) )
        self.assertEqual( block_list[1].loc, (2, 3) )
        x = [ f1 == f2 for f1, f2 in zip( block_list[0].get_fun_vals(), fun_vals[0] ) ]
        y = [ f1 == f2 for f1, f2 in zip( block_list[1].get_fun_vals(), fun_vals[1] ) ]
        self.assertTrue( all( x ) )
        self.assertTrue( all( y ) )


if __name__ == '__main__':
    tf.test.main()
