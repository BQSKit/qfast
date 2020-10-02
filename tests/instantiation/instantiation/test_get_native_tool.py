import tensorflow as tf
import numpy      as np

import qfast
from qfast import list_native_tools, get_native_tool


class TestGetNativeTool ( tf.test.TestCase ):

    def test_get_native_tool_invalid ( self ):
        native_tools = list_native_tools()

        invalid_tool = "test_dummy_tool"

        while invalid_tool in native_tools:
            invalid_tool += "a"

        self.assertRaises( ValueError, get_native_tool, invalid_tool )

    def test_get_native_tool_valid ( self ):
        kak = get_native_tool( "kak" )
        self.assertTrue( kak == qfast.native.kak )


if __name__ == '__main__':
    tf.test.main()