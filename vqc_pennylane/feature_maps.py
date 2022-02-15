# Feature map architectures implemented in pennylane, to be used in buidling
# the trainable variational quantum circuit. The feature maps are responsible
# with loading the data into the circuit and do not learn during training.
import pennylane as pnl
import numpy as np

from itertools import combinations


def zzfm(nqubits, inputs):
    """
    ZZfeatureMap pennylane implementation. The building of this circuit is
    done inside the pennylane namespace, so no need to return anything.
    @nqubits :: The number of qubits to use.
    @inputs  :: Mapping of inputs into the qubits.
    """
    for x in range(len(inputs)):
        pnl.Hadamard(x)
        pnl.RZ(2.0 * inputs[x], wires=x)

    for qpair in list(combinations(range(len(inputs)), 2)):
        pnl.CZ(wires=[qpair[0], qpair[1]])
        pnl.RZ(
            2.0 * (np.pi - inputs[qpair[0]]) * (np.pi - inputs[qpair[1]]),
            wires=qpair[1],
        )
        pnl.CZ(wires=[qpair[0], qpair[1]])
