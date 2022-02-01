import pennylane as pnl
import numpy as np

def zzfm(nqubits, inputs):
    """
    ZZfeatureMap pennylane implementation. The building of this circuit is
    done inside the pennylane namespace, so no need to return anything.
    @nqubits :: The number of qubits to use.
    @inputs  :: Mapping of inputs into the qubits.
    """
    for idx in range(len(inputs)):
        pnl.Hadamard(idx)
        pnl.RZ(2.0*inputs[idx], wires=idx)

    for qpair in list(combinations(range(len(inputs), 2))):
        pnl.CZ(qpair[0], qpair[1])
        pnl.RZ(2.0*(np.pi - inputs[qpair[0]])*(np.pi - inputs[qpair[1]])
        pnl.CZ(qpair[0], qpair[1])
