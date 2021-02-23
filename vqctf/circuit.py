import pennylane as qml
import numpy as np

import vqctf.zzlane as fmap
import vqctf.twolane as vform

nqubits = 4

circuit_desc = "4-qubits\nZZ(4) 2L(3rep) ZZ(4) 2L(3rep)"

dev = qml.device("default.qubit", wires=nqubits)

state_0 = [[1], [0]]
state_1 = [[0], [1]]
states = [state_0, state_1]

def density_matrix(state):
	return state * np.conj(state).T

y = density_matrix(state_0)

@qml.qnode(dev, interface="tf")
def qcircuit(inputs, theta):
	fmap.get_circuit(nqubits, inputs[0:4])
	vform.get_circuit(nqubits, theta[0:16], reps = 3)
	fmap.get_circuit(nqubits, inputs[4:8])
	vform.get_circuit(nqubits, theta[16:32], reps = 3)
	return qml.expval(qml.Hermitian(y, wires=[0]))





