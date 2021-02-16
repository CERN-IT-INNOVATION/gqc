import pennylane as qml
import numpy as np

import vqctf.zzlane as fmap
import vqctf.twolane as vform

nqubits = 4

circuit_desc = "4-qubits\nZZ(4) 2L(2rep) ZZ(4) 2L(2rep)"

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
	vform.get_circuit(nqubits, theta[0:12])
	fmap.get_circuit(nqubits, inputs[4:8])
	vform.get_circuit(nqubits, theta[12:24])
	return qml.expval(qml.Hermitian(y, wires=[0]))

def test(theta, data):
	predicted = []
	for i in range(len(data)):
		p0 = prob0(theta, data(i))
		if (p0 >= .5):
			predicted.append(0)
		else:
			predicted.append(1)
	return np.array(predicted)



