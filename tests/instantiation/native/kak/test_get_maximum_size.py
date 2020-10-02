import numpy    as np
import unittest as ut

from qfast.instantiation.native.kak import KAKTool


class TestKakGetMaximumSize ( ut.TestCase ):

    def test_kak_get_maximum_size ( self ):
        size = KAKTool().get_maximum_size()
        self.assertEqual( size, 2 )


if __name__ == '__main__':
    ut.main()

