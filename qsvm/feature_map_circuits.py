# Module where all tested quantum circuits are defined using qiskit.
# To be used for the classifiers and expressibility and entanglement studies.

from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np

def u2Reuploading(nqubits=8, nfeatures=16) -> QuantumCircuit:
	"""
	Constructs the u2Reuploading feature map.
	@nqubits   :: Int number of qubits used.
	@nfeatures :: Number of variables in the dataset to be processed.

	returns :: The quantum circuit object form qiskit.
	"""
	x = ParameterVector('x', nfeatures)
	qc = QuantumCircuit(nqubits)
	for feature, qubit in zip(range(0, 2*nqubits, 2), range(nqubits)):
		qc.u(np.pi/2, x[feature], x[feature+1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
	for i in range(nqubits):
		if i == nqubits-1:
			break
		qc.cx(i, i+1)
	for feature, qubit in zip(range(2*nqubits, nfeatures, 2), range(nqubits)):
		qc.u(np.pi/2, x[feature], x[feature+1], qubit)

	for feature, qubit in zip(range(0, 2*nqubits, 2), range(nqubits)):
		qc.u(x[feature], x[feature+1], 0, qubit)

	return qc

