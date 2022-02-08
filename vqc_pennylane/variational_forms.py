# The variational forms used in building the variational quantum circuit.
# These parts of the circuit contain weights which are updated during
# the training process.
import pennylane as pnl
import numpy as np

from itertools import combinations


def vforms_weights(vform_choice, repeats, nqubits):
    """
    Returns the number of weights a certain circuit has with 1 repeat and
    without any trailing gates.
    @vform_choice :: String of the choice of variational form.
    """
    switcher = {
        "two_local": lambda : nqubits*(repeats+1) ,
    }

    nweights = switcher.get(vform_choice, lambda: None)()
    if nweights is None:
        raise TypeError("Specified variational form is not implemented!")

    return nweights

def two_local(nqubits, weights, repeats=1, entanglement="linear"):
    """
    Pennylane implementation of the twolocal variational form.
    Do not need to return anything since the circuit is saved to the namespace.
    @nqubits      :: The number of qubits to use.
    @weights      :: Weights to be adjusted in the training.
    @repeats      :: Number of times to repeat consecutively the vform.
    @entanglement :: The type of entanglement (structure of the cnot gates).
    """
    for repeat in range(repeats):
        for qubit in range(nqubits):
            pnl.RY(weights[repeat][qubit], wires=qubit)
        if entanglement == "linear":
            two_local_linear(nqubits)
        elif entanglement == "full":
            two_local_full(nqubits)
        else:
            raise AttributeError("Unknown entanglement pattern!")

    for qubit in range(nqubits):
        pnl.RY(weights[repeat][qubit], wires=qubit)

def two_local_linear(nqubits):
    """
    Part of the twolocal circuit, with linear entanglement.
    @nqubits      :: The number of qubits to use.
    """
    for qubit in range(nqubits-1):
        pnl.CNOT(wires=[qubit, qubit+1])

def two_local_full(nqubits):
    """
    Part of the twolocal circuit, fully entangled.
    @nqubits      :: The number of qubits to use.
    """
    for qpair in list(combinations(range(len(nqubits)), 2)):
        pnl.CNOT(wires=[qpair[0], qpair[1]])
