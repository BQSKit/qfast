import numpy    as np
import unittest as ut

from qfast.instantiation.native.qs import QSearchTool


class TestQSGetMaximumSize ( ut.TestCase ):

    def test_qs_get_maximum_size ( self ):
        qtool = QSearchTool()
        block_size = qtool.get_maximum_size()
        self.assertEqual( block_size, 3 )


if __name__ == '__main__':
    ut.main()
