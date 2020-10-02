import tensorflow as tf
import numpy      as np

from qfast import list_native_tools


class TestListNativeTools ( tf.test.TestCase ):

    def test_list_native_tools ( self ):
        native_tools = list_native_tools()
        self.assertTrue( "kak" in native_tools )


if __name__ == '__main__':
    tf.test.main()