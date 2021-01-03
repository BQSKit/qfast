"""
This module implements QSearch's Leap Compiler
as a native tool plugin to QFAST.
"""

import qsearch
from qsearch import options, assemblers, leap_compiler, post_processing

from qfast import utils
from qfast import perm
from qfast.instantiation import nativetool


class QSearchTool ( nativetool.NativeTool ):
    """Synthesize tool built on QSearch's Leap Compiler."""

    def get_maximum_size ( self ):
        """
        The maximum size of a unitary matrix (in qubits) that can be
        decomposed with this tool.

        Returns:
            (int): The qubit count this tool can handle.
        """

        return 3

    def map_basis_str_to_gateset ( self, basis_gates ):
        """Converts the string descriptor to a qsearch basis gate object."""
        if basis_gates is None or "cx" in basis_gates:
            return qsearch.gatesets.QubitCNOTLinear()
        elif "cz" in basis_gates:
            return qsearch.gatesets.QubitCZLinear()
        elif "iswap" in basis_gates:
            return qsearch.gatesets.QubitISwapLinear()
        elif "rxx" in basis_gates:
            return qsearch.gatesets.QubitXXLinear()
        else:
            raise ValueError( "Unsupported basis gates: %s" % basis_gates )

    def synthesize ( self, utry, **kwargs ):
        """
        Synthesis function with this tool.

        Args:
            utry (np.ndarray): The unitary to synthesize.

        Returns
            qasm (str): The synthesized QASM output.

        Raises:
            TypeError: If utry is not a valid unitary.

            ValueError: If the utry has invalid dimensions.
        """


        if not utils.is_unitary( utry, tol = 1e-14 ):
            raise TypeError( "utry must be a valid unitary." )

        if utry.shape[0] > 2 ** self.get_maximum_size():
            raise ValueError( "utry has incorrect dimensions." )

        # Parse kwargs
        basis_gates = [ "cx" ]
        coupling_graph = [ (0, 1), (1, 2) ]
        if "basis_gates" in kwargs:
            basis_gates = kwargs["basis_gates"] or basis_gates
        if "coupling_graph" in kwargs:
            coupling_graph = kwargs["coupling_graph"] or coupling_graph

        # Prepermute unitary to line up coupling_graph
        # This is done because qsearch handles pure linear topologies best
        if utils.get_num_qubits( utry ) == 3:
            a = (0, 1) in coupling_graph
            b = (1, 2) in coupling_graph
            c = (0, 2) in coupling_graph

            if not (a and b):
                if (a and c):
                    # Permute 0 and 1
                    P = perm.calc_permutation_matrix( 3, (1, 0, 2) )
                    utry = P @ utry @ P.T
                elif (b and c):
                    # Permute 1 and 2
                    P = perm.calc_permutation_matrix( 3, (0, 2, 1) )
                    utry = P @ utry @ P.T
                else:
                    raise ValueError( "Invalid coupling graph." )


        # Pass options into qsearch, being maximally quiet,
        # and set the target to utry
        opts = options.Options()
        opts.target = utry
        opts.gateset = self.map_basis_str_to_gateset( basis_gates )
        opts.verbosity = 0
        opts.write_to_stdout = False
        opts.reoptimize_size = 7

        # use the LEAP compiler, which scales better than normal qsearch
        compiler = leap_compiler.LeapCompiler()
        output = compiler.compile( opts )

        # LEAP requires some post-processing
        pp = post_processing.LEAPReoptimizing_PostProcessor()
        output = pp.post_process_circuit( output, opts )
        output = assemblers.ASSEMBLER_IBMOPENQASM.assemble( output )

        # Renumber qubits in circuit if we flipped the unitary
        if utils.get_num_qubits( utry ) == 3:
            a = (0, 1) in coupling_graph
            b = (1, 2) in coupling_graph
            c = (0, 2) in coupling_graph

            if not (a and b):
                if (a and c):
                    # Permute 0 and 1
                    str0 = "[0]"
                    str1 = "[1]"
                        
                elif (b and c):
                    # Permute 1 and 2
                    str0 = "[1]"
                    str1 = "[2]"
 
                output = output.replace( str0, "[tmp]" )
                output = output.replace( str1, str0 )
                output = output.replace( "[tmp]", str1 )

        return output

