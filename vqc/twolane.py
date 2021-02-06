import pennylane as qml
from itertools import combinations
import numpy as np

description = "Two-local VF with RY CX PL v1"

def get_circuit(qubits, theta, reps = 2):
	
	for r in range(reps):
		for i in range(qubits):
			qml.RY(theta[r * qubits + i], wires = i)
		for i in range(qubits - 1):
			qml.CNOT(wires = [i, i + 1])
	
	for i in range(qubits):
		qml.RY(theta[reps * qubits + i], wires = i)

			
"""
def my_circuit(x):
	get_circuit(4, x)
	return qml.expval(qml.PauliZ(0))



dev = qml.device('default.qubit', wires=4)
circuit = qml.QNode(my_circuit, dev)

circuit([1,2,3,4,5,6,7,8,9,10,11,12])

print(circuit.draw())
"""

