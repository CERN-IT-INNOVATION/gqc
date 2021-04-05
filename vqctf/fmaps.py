import pennylane as qml
from itertools import combinations
import numpy as np

def zzfm(qubits, x, scaled = False):
	
	features = len(x);

	last = 0 # The last feature that we have loaded.
	
	while (last < features):
		# Number of features that we will load.
		nload = min(features - last, qubits) 

		for i in range(nload):
			qml.Hadamard(i)
			if scaled:
				qml.RZ(2.0 * np.pi * x[last + i], wires = i)
			else:
				qml.RZ(2.0*x[last + i], wires = i)

		for pair in list(combinations(range(nload),2)):
			a = pair[0]
			b = pair[1]
			if (b < a): # Just in case...
				tmp = b
				b = a
				a = tmp
			qml.CZ(wires = [a, b])
			if scaled:
				qml.RZ(2.0 * np.pi * (1 - x[last + a]) * (1 - x[last + b]), wires = b)
			else:
				qml.RZ(2.0 * (np.pi - x[last + a]) * (np.pi - x[last + b]), wires = b)
			qml.CZ(wires = [a,b])

		last += nload;
