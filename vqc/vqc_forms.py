# VQC forms to be used in building the circuit.
import pennylane as qml
import numpy as np
from itertools import combinations


def twolocal_full(qubits, theta, repeats=2):
    """
    Twolocal form with full entanglement between qubits.
    @qubits  :: Int the number of qubits in the form.
    @theta   :: List of the encoding angles.
    @repeats :: Int of how many times should this form be repeated.
    """
    for repeat in range(repeats):
        for qubit in range(qubits):
            qml.RY(theta[repeats * qubits + qubit], wires=qubit)

        for qubit_pair in list(combinations(range(qubits), 2)):
            if (qubit_pair[1] < qubit_pair[0]):
                tmp = qubit_pair[1]
                qubit_pair[1] = qubit_pair[0]
                qubit_pair[0] = tmp

            qml.CNOT(wires=[qubit_pair[0], qubit_pair[1]])


def twolocal_linear(qubits, theta, repeats=2):
    """
    The twolocal form with linear entanglement between qubits.
    @qubits  :: Int the number of qubits in the form.
    @theta   :: List of the encoding angles.
    @repeats :: Int of how many times should this form be repeated.
    """
    for repeat in range(repeats):
        for qubit in range(qubits):
            qml.RY(theta[repeats * qubits + qubit], wires=qubit)
        for qubit in range(qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

    for i in range(qubits):
        qml.RY(theta[repeats * qubits + i], wires=i)


def zzfm(qubits, x):
    """
    ZZ feature map.
    @qubits :: Int the number of qubits in the form.
    @x      :: No idea what this is since it is assigned from the namespace.
    """
    features = len(x)
    last_feature = 0
    
    while last_feature < features:
        nload = min(features - last_feature, qubits)
        for idx in range(nload):
            qml.Hadamard(idx)
            qml.RZ(2.0*x[last_feature + idx], wires=idx)

        for pair in list(combinations(range(nload), 2)):
            if pair[1] < pair[0]:
                tmp = pair[1]
                pair[1] = pair[0]
                pair[0] = tmp

            qml.CZ(wires=[pair[0], pair[1]])
            qml.RZ(2.0 * (np.pi - x[last_feature + pair[0]]) * (np.pi -
                   x[last_feature + pair[1]]), wires=pair[1])
            qml.CZ(wires=[pair[0], pair[1]])

        last_feature += nload


def zzfm_scaled(qubits, x):
    """
    ZZ feature map.
    @qubits :: Int the number of qubits in the form.
    @x      ::
    """
    features = len(x)
    last_feature = 0

    while last_feature < features:
        # Number of features that we will load.
        nload = min(features - last_feature, qubits)

        for i in range(nload):
            qml.Hadamard(i)
            qml.RZ(2.0 * np.pi * x[last_feature + i], wires=i)

        for pair in list(combinations(range(nload), 2)):
            a = pair[0]
            b = pair[1]
            if (b < a):
                tmp = b
                b = a
                a = tmp
            qml.CZ(wires=[a, b])
            qml.RZ(2.0 * np.pi * (1 - x[last_feature + a]) * (1 -
                   x[last_feature + b]), wires=b)
            qml.CZ(wires=[a, b])

        last_feature += nload
