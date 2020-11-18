# QFAST: Quantum Fast Approximate Synthesis Tool

QFAST is a quantum synthesis tool designed to produce short circuits and to scale well in practice. QFAST uses a mathematical model of circuits encoding both gate placement and function. This is packaged together with a hierarchical stochastic gradient descent formulation that combines “coarse-grained” fast optimization during circuit structure search with a better, but slower, stage only in the final circuit refinement.

## Installation

The best way to install this python package is with pip.

```
pip install qfast
```

## Usage

QFAST can be used to convert a quantum operation specified by a unitary matrix into a circuit given by [openqasm](https://github.com/Qiskit/openqasm) code. There is a command-line interface provided with qfast that can be accessed by `python -m qfast`. This can be used to synthesize a matrix.

```
python -m qfast input.unitary output.qasm
```

Here the `input.unitary` file is a NumPy matrix saved with [np.savetxt](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html), the qasm output will be saved in the `output.qasm` file and the KAK native tool will be used. The command-line help option `python -m qfast -h` can be used for further information.

QFAST can also be used as a library, [an example](https://github.com/BQSKit/qfast/blob/master/examples/synthesize_qft4.py) is included.

### Native Tools

Native tools are necessary for QFAST to perform instantiation. During decomposition, the input unitary matrix is hierarchically broken into many smaller unitaries. At some level in the hierarchy, QFAST switches to instantiation, which uses a native synthesis tool to convert the small unitaries into native gates.

Included with this python package is the QSearch native tool. Here are some others:

- [qfast-qiskit](https://github.com/BQSKit/qfast-qiskit): Several qiskit tools
- [qfast-uq](https://github.com/BQSKit/qfast-uq): A UniversalQCompiler native tool (deprecated)
- [qfast-qs](https://github.com/BQSKit/qfast-qs): A QSearch native tool (Now default)

## References

Younis, Ed, et al. "[QFAST: Quantum Synthesis Using a Hierarchical Continuous Circuit Space.](https://arxiv.org/abs/2003.04462)" arXiv preprint arXiv:2003.04462 (2020).

## Copyright

Quantum Fast Approximate Synthesis Tool (QFAST) Copyright (c) 2020,
The Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
