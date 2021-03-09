import pennylane as qml
import numpy as np

def twolocal(qubits, theta, reps = 2):
	
	for r in range(reps):
		for i in range(qubits):
			qml.RY(theta[r * qubits + i], wires = i)
		for i in range(qubits - 1):
			qml.CNOT(wires = [i, i + 1])
	
	for i in range(qubits):
		qml.RY(theta[reps * qubits + i], wires = i)
