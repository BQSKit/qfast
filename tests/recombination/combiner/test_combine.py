import numpy    as np
import unittest as ut

from qfast.recombination.combiner import Combiner

class TestCombinerCombine ( ut.TestCase ):

    def test_combiner_combine_invalid_type ( self ):
        combiner = Combiner()

        qasm_list_0 = 1
        qasm_list_1 = "a"
        qasm_list_2 = [ 1 ]
        qasm_list_3 = [ ( 1, 1 ) ]
        qasm_list_4 = [ ( 1, ( 1 ) ) ]
        qasm_list_5 = [ ( 1, "a" ) ]
        qasm_list_6 = [ 1 ]
        qasm_list_7 = [ ("a", "b") ]

        self.assertRaises( TypeError, combiner.combine, qasm_list_0 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_1 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_2 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_3 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_4 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_5 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_6 )
        self.assertRaises( TypeError, combiner.combine, qasm_list_7 )

    def test_combiner_combine_invalid_value ( self ):
        combiner = Combiner()

        loc = (0, 1)

        loc_fixed = [ loc ]

        qasm1 = ( "ERROR_TEST", loc )
        qasm2 = ( ( "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[7];\n"
                    "cx q[0],q[1];\n" ), loc )
        qasm3 = ( ( "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[2];\n"
                    "h q[0];\n" ), loc )

        for qasm_list in [ [ qasm1 ], [ qasm2 ], [ qasm3 ] ]:
            self.assertRaises( ValueError, combiner.combine, qasm_list )

    def test_combiner_combine_basic_2 ( self ):
        combiner = Combiner()

        qasm = ( "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "cx q[0],q[1];\n" )

        loc = (0, 1)

        qasm_list = [ ( qasm, loc ) ]

        out_qasm = combiner.combine( qasm_list )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[2];\n"
                                      "cx q[0],q[1];\n" ) )

    def test_combiner_combine_basic_3 ( self ):
        combiner = Combiner()

        qasm = ( "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "cx q[0],q[1];\n" )

        loc = (0, 2)

        qasm_list = [ ( qasm, loc ) ]

        out_qasm = combiner.combine( qasm_list )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[3];\n"
                                      "cx q[0],q[2];\n" ) )

        loc = (1, 2)

        qasm_list = [ ( qasm, loc ) ]

        out_qasm = combiner.combine( qasm_list )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[3];\n"
                                      "cx q[1],q[2];\n" ) )

    def test_combiner_combine_adv_1 ( self ):
        combiner = Combiner()

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

        qasm_list = list( zip( qasm_list, loc_fixed ) )

        out_qasm = combiner.combine( qasm_list )

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

    def test_combiner_combine_rxryrz ( self ):
        combiner = Combiner()

        qasm_list = [ ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[0],q[1];\n"
                        "cx q[1],q[2];\n"
                        "cx q[2],q[3];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "rz(1.57) q[0];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[2],q[3];\n"
                        "cx q[1],q[2];\n"
                        "cx q[0],q[1];\n" ) ]

        loc_fixed = [ (0, 1, 2, 3), (3,), (0, 1, 2, 3) ]

        qasm_list = list( zip( qasm_list, loc_fixed ) )

        out_qasm = combiner.combine( qasm_list )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[4];\n"
                                      "cx q[0],q[1];\n"
                                      "cx q[1],q[2];\n"
                                      "cx q[2],q[3];\n"
                                      "u1(1.57) q[3];\n"
                                      "cx q[2],q[3];\n"
                                      "cx q[1],q[2];\n"
                                      "cx q[0],q[1];\n" ) )

        qasm_list = [ ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[0],q[1];\n"
                        "cx q[1],q[2];\n"
                        "cx q[2],q[3];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "ry(1.57) q[0];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[2],q[3];\n"
                        "cx q[1],q[2];\n"
                        "cx q[0],q[1];\n" ) ]

        loc_fixed = [ (0, 1, 2, 3), (3,), (0, 1, 2, 3) ]

        qasm_list = list( zip( qasm_list, loc_fixed ) )

        out_qasm = combiner.combine( qasm_list )

        self.assertEqual( out_qasm, ( "OPENQASM 2.0;\n"
                                      "include \"qelib1.inc\";\n"
                                      "qreg q[4];\n"
                                      "cx q[0],q[1];\n"
                                      "cx q[1],q[2];\n"
                                      "cx q[2],q[3];\n"
                                      "u3(1.57,0.0,0.0) q[3];\n"
                                      "cx q[2],q[3];\n"
                                      "cx q[1],q[2];\n"
                                      "cx q[0],q[1];\n" ) )

        qasm_list = [ ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[0],q[1];\n"
                        "cx q[1],q[2];\n"
                        "cx q[2],q[3];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "ry(1.57) q[0];\n"
                        "rx(1.57) q[0];\n"
                        "rz(1.57) q[0];\n" ),
                      ( "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[4];\n"
                        "cx q[2],q[3];\n"
                        "cx q[1],q[2];\n"
                        "cx q[0],q[1];\n" ) ]

        loc_fixed = [ (0, 1, 2, 3), (3,), (0, 1, 2, 3) ]

        qasm_list = list( zip( qasm_list, loc_fixed ) )

        out_qasm = combiner.combine( qasm_list )

        self.assertTrue( "OPENQASM 2.0;\n"
                         "include \"qelib1.inc\";\n"
                         "qreg q[4];\n"
                         "cx q[0],q[1];\n"
                         "cx q[1],q[2];\n"
                         "cx q[2],q[3];\n" in out_qasm )

        self.assertTrue( "cx q[2],q[3];\n"
                         "cx q[1],q[2];\n"
                         "cx q[0],q[1];\n" in out_qasm )

        self.assertTrue( "u3" in out_qasm )


if __name__ == '__main__':
    ut.main()

