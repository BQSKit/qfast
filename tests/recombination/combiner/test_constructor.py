import numpy    as np
import unittest as ut

from qfast.recombination.combiner import Combiner

class TestCombinerConstructor ( ut.TestCase ):

    def test_combiner_constructor ( self ):
        combiner = Combiner()
        self.assertTrue( combiner.optimization )

        combiner = Combiner( False )
        self.assertFalse( combiner.optimization )


if __name__ == '__main__':
    ut.main()

