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
python -m qfast --unitary-file=input.unitary --qasm-file=output.qasm --native-tool=kak
```

Here the `input.unitary` file is a NumPy matrix saved with [np.savetxt](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html), the qasm output will be saved in the `output.qasm` file and the KAK native tool will be used. The command-line help option `python -m qfast -h` can be used for further information.

QFAST can also be used as a library, [an example](https://github.com/edyounis/qfast/blob/master/examples/synthesize_qft4.py) is included.

### Native Tools

Native tools are necessary for QFAST to perform instantiation. During decomposition, the input unitary matrix is hierarchically broken into many smaller unitaries. At some level in the hierarchy, QFAST switches to instantiation using a native synthesis tool to perform the final step.

Included with this python package is the KAK native tool. Here are some others:

- [qfast-uq](https://github.com/edyounis/qfast-uq): A UniversalQCompiler native tool

## References

Younis, Ed, et al. "[QFAST: Quantum Synthesis Using a Hierarchical Continuous Circuit Space.](https://arxiv.org/abs/2003.04462)" arXiv preprint arXiv:2003.04462 (2020).