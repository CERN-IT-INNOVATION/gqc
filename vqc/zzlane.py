import pennylane as qml
from itertools import combinations
import numpy as np

description = "Custom ZZ feature map for PL v1"

def get_circuit(qubits, x):
	
	features = len(x);

	last = 0 # The last feature that we have loaded.
	
	while (last < features):
		# Number of features that we will load.
		nload = min(features - last, qubits) 

		for i in range(nload):
			qml.Hadamard(i)
			qml.RZ(2.0*x[last + i], wires = i)

		for pair in list(combinations(range(nload),2)):
			a = pair[0]
			b = pair[1]
			if (b < a): # Just in case...
				tmp = b
				b = a
				a = tmp
			qml.CZ(wires = [a, b])
			qml.RZ(2.0 * (np.pi - x[last + a].val) * (np.pi - x[last + b].val), wires = b)
			qml.CZ(wires = [a,b])

		last += nload;
	

"""
def my_circuit(x):
	get_circuit(4, x)
	return qml.expval(qml.Hermitian([0,1,0],wires=[0,1,2]))



dev = qml.device('default.qubit', wires=4)
circuit = qml.QNode(my_circuit, dev)

circuit([1,2,3,4,5,6])

print(circuit.draw())
"""

