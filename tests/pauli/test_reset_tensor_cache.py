import tensorflow as tf
import numpy      as np

from qfast import get_norder_paulis_tensor, reset_tensor_cache
from qfast.pauli import norder_paulis_tensor_map


class TestResetTensorCache ( tf.test.TestCase ):

    def test_reset_tensor_cache ( self ):
        pauli_tensors = get_norder_paulis_tensor( 4 )
        self.assertTrue( len( norder_paulis_tensor_map ) >= 5 )
        reset_tensor_cache()
        print( len( norder_paulis_tensor_map ) )
        self.assertTrue( len( norder_paulis_tensor_map ) == 2 )


if __name__ == '__main__':
    tf.test.main()
