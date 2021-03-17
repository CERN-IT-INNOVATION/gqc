import pennylane as qml
import numpy as np
from itertools import combinations

def twolocal(qubits, theta, reps = 2, entanglement = "linear"):
	
	for r in range(reps):
		for i in range(qubits):
			qml.RY(theta[r * qubits + i], wires = i)
		if entanglement == "linear":
			for i in range(qubits - 1):
				qml.CNOT(wires = [i, i + 1])
		elif entanglement == "full":
			for pair in list(combinations(range(qubits),2)):
				a = pair[0]
				b = pair[1]
				if (b < a): # Just in case...
					tmp = b
					b = a
					a = tmp
				qml.CNOT(wires = [a, b])
		else:
			raise Exception("Unknown entanglement pattern")

	
	for i in range(qubits):
		qml.RY(theta[reps * qubits + i], wires = i)
