import tensorflow as tf
import numpy      as np

from qfast import recombination


class TestRecombination ( tf.test.TestCase ):

    def test_recombination_invalid_type ( self ):
        qasm = ( "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "cx q[0],q[1];\n" )

        loc = (0, 1)

        qasm_list = [ qasm ]
        loc_fixed = [ loc ]

        qasm_list_1 = [ 1 ]
        loc_fixed_1 = [ 1 ]
        loc_fixed_2 = [ ("a", "b") ]
        loc_fixed_3 = [ (1, 1) ]

        self.assertRaises( TypeError, recombination, 1, loc_fixed )
        self.assertRaises( TypeError, recombination, qasm_list, 1 )
        self.assertRaises( TypeError, recombination, 1, 1 )
        self.assertRaises( TypeError, recombination, qasm_list_1, loc_fixed )
        self.assertRaises( TypeError, recombination, qasm_list, loc_fixed_1 )
        self.assertRaises( TypeError, recombination, qasm_list, loc_fixed_2 )
        self.assertRaises( TypeError, recombination, qasm_list, loc_fixed_3 )

    def test_recombination_invalid_value ( self ):
        loc = (0, 1)

        loc_fixed = [ loc ]

        qasm1 = "ERROR_TEST"
        qasm2 = ( "OPENQASM 2.0;\n"
                  "include \"qelib1.inc\";\n"
                  "qreg q[7];\n"
                  "cx q[0],q[1];\n" )
        qasm3 = ( "OPENQASM 2.0;\n"
                  "include \"qelib1.inc\";\n"
                  "qreg q[2];\n"
                  "h q[0];\n" )

        for qasm_list in [ [ qasm1 ], [ qasm2 ], [ qasm3 ] ]:
            self.assertRaises( ValueError, recombination, qasm_list, loc_fixed )

    def test_recombination_basic_2 ( self ):
        qasm = ( "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "cx q[0],q[1];\n" )

        loc = (0, 1)

        qasm_list = [ qasm ]
        loc_fixed = [ loc ]

        out_qasm = recombination( qasm_list, loc_fixed )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[2];\n"
                                      "cx q[0],q[1];\n" ) )

    def test_recombination_basic_3 ( self ):
        qasm = ( "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "cx q[0],q[1];\n" )

        loc = (0, 2)

        qasm_list = [ qasm ]
        loc_fixed = [ loc ]

        out_qasm = recombination( qasm_list, loc_fixed )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[3];\n"
                                      "cx q[0],q[2];\n" ) )

        loc = (1, 2)

        qasm_list = [ qasm ]
        loc_fixed = [ loc ]

        out_qasm = recombination( qasm_list, loc_fixed )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[3];\n"
                                      "cx q[1],q[2];\n" ) )

    def test_recombination_adv_1 ( self ):
        qasm_list = [ ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[0],q[1];\n"
                        "cx q[1],q[2];\n"
                        "cx q[2],q[3];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "u3(0.11,1.57,4.71) q[0];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[2],q[3];\n"
                        "cx q[1],q[2];\n"
                        "cx q[0],q[1];\n" ) ]

        loc_fixed = [ (0, 1, 2, 3), (3,), (0, 1, 2, 3) ]

        out_qasm = recombination( qasm_list, loc_fixed )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[4];\n"
                                      "cx q[0],q[1];\n"
                                      "cx q[1],q[2];\n"
                                      "cx q[2],q[3];\n"
                                      "u3(0.11,1.57,4.71) q[3];\n"
                                      "cx q[2],q[3];\n"
                                      "cx q[1],q[2];\n"
                                      "cx q[0],q[1];\n" ) )
