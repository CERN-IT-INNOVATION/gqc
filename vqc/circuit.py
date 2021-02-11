import pennylane as qml
import numpy as np

import vqc.zzlane as fmap
import vqc.twolane as vform

nqubits = 4


dev = qml.device("default.qubit", wires=nqubits)

@qml.qnode(dev, interface="autograd")
def qcircuit(theta, data, y):
	fmap.get_circuit(nqubits, data[0:4])
	vform.get_circuit(nqubits, theta[0:12])
	fmap.get_circuit(nqubits, data[4:8])
	vform.get_circuit(nqubits, theta[12:24])
	return qml.expval(qml.Hermitian(y, wires=[0]))



state_0 = [[1], [0]]
state_1 = [[0], [1]]
states = [state_0, state_1]

def density_matrix(state):
	return state * np.conj(state).T


def cost(theta, data, labels):
	# Compute prediction for each input in data batch
	loss = 0.0
	dm_states= [density_matrix(s) for s in states]
	for i in range(len(data)):
		f = qcircuit(theta, data[i], dm_states[labels[i]])
		loss = loss + (1 - f) ** 2
	return loss / len(data)


def test(theta, data):
	fidelity_values = []
	dm_states = [density_matrix(s) for s in states]
	predicted = []

	for i in range(len(data)):
		fidel_function = lambda y: qcircuit(theta, data[i], y)
		fidelities = [fidel_function(dm) for dm in dm_states]
		best_fidel = np.argmax(fidelities)

		predicted.append(best_fidel)
		fidelity_values.append(fidelities)

	return np.array(predicted), np.array(fidelity_values)


def accuracy_score(label_true, label_pred):
	score = label_true == label_pred
	return score.sum() / len(label_true)


def iterate_minibatches(inputs, targets, batch_size):
	for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
		idxs = slice(start_idx, start_idx + batch_size)
		yield inputs[idxs], targets[idxs]





