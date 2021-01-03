import numpy    as np
import unittest as ut

from qfast.topology import Topology
from qfast.decomposition.decomposer import Decomposer

class TestDecomposerConstructor ( ut.TestCase ):

    def test_decomposer_constructor_invalid ( self ):
        valid_utry = np.identity( 8 )
        valid_target_gate_size = 2
        valid_model = "PermModel"
        valid_optimizer = "LBFGSOptimizer"
        valid_hierarchy_fn = lambda x : 2

        invalid_utry = "a"
        invalid_target_gate_size_1 = -1
        invalid_target_gate_size_2 = 1000
        invalid_target_gate_size_3 = "a"
        invalid_model = "test_dummy_model"
        invalid_optimizer = "test_dummy_optimizer"
        invalid_hierarchy_fn = "a"
        invalid_topology = "a"

        self.assertRaises( TypeError, Decomposer, invalid_utry )

        self.assertRaises( ValueError, Decomposer, valid_utry,
                           invalid_target_gate_size_1 )
        self.assertRaises( ValueError, Decomposer, valid_utry,
                           invalid_target_gate_size_2 )
        self.assertRaises( TypeError, Decomposer, valid_utry,
                           invalid_target_gate_size_3 )

        self.assertRaises( RuntimeError, Decomposer, valid_utry,
                           valid_target_gate_size, invalid_model )

        self.assertRaises( RuntimeError, Decomposer, valid_utry,
                           valid_target_gate_size, valid_model,
                           invalid_optimizer )

        self.assertRaises( TypeError, Decomposer, valid_utry,
                           valid_target_gate_size, valid_model,
                           valid_optimizer, invalid_hierarchy_fn )

        self.assertRaises( TypeError, Decomposer, valid_utry,
                           valid_target_gate_size, valid_model,
                           valid_optimizer, valid_hierarchy_fn,
                           invalid_topology )

    def test_decomposer_constructor_valid ( self ):
        valid_utry = np.identity( 8 )
        valid_target_gate_size = 2
        valid_model = "PermModel"
        valid_optimizer = "LBFGSOptimizer"
        valid_hierarchy_fn = lambda x : 2
        valid_topology  = Topology( 3, [ (0, 1), (1, 2) ] )

        decomposer = Decomposer( valid_utry, valid_target_gate_size,
                                 valid_model, valid_optimizer,
                                 valid_hierarchy_fn, valid_topology )
        
        self.assertTrue( np.allclose( decomposer.utry, valid_utry ) )
        self.assertTrue( decomposer.num_qubits == 3 )
        self.assertTrue( decomposer.target_gate_size == 2 )
        self.assertTrue( decomposer.hierarchy_fn(2112) == 2 )
        
        model = decomposer.model.__name__
        optimizer = decomposer.optimizer.__name__

        self.assertTrue( model == valid_model )
        self.assertTrue( optimizer == valid_optimizer )

        for link in valid_topology.coupling_graph:
            self.assertTrue( link in decomposer.topology.coupling_graph )

        self.assertTrue( len( decomposer.topology.coupling_graph ) == 2 )


if __name__ == '__main__':
    ut.main()

