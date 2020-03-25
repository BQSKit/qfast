import tensorflow as tf
import numpy      as np

from qfast.native.kak import get_native_block_size


class TestKakGetNativeBlockSize ( tf.test.TestCase ):

    def test_kak_get_native_block_size ( self ):
        block_size = get_native_block_size()
        self.assertEqual( block_size, 2 )


if __name__ == '__main__':
    tf.test.main()
