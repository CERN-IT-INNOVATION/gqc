from qiskit.circuit import ParameterVector, QuantumCircuit
from itertools import combinations
import math

description = "Custom ZZ feature map v1"

def get_circuit(qubits, features):
	
	if (qubits > features):
		raise Exception("Why would you have more qubits than features!?")

	x = ParameterVector('x', features)
	qc = QuantumCircuit(qubits)

	last = 0 # The last feature that we have loaded.
	
	while (last < features):
		# Number of features that we will load.
		nload = min(features - last, qubits) 

		for i in range(nload):
			qc.h(i)
			qc.p(2.0*x[last + i], i)

		for pair in list(combinations(range(nload),2)):
			a = pair[0]
			b = pair[1]
			if (b < a): # Just in case...
				tmp = b
				b = a
				a = tmp
			qc.cx(a, b)
			qc.p(2.0 * (math.pi - x[last + a]) * (math.pi - x[last + b]), b)
			qc.cx(a,b)
	
		last += nload;
	
	return qc

