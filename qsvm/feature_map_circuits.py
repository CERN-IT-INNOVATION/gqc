'''
Module where all tested quantum circuits are defined using qiskit.
To be used for the classifiers and expressibility and entanglement studies.
'''

# Amplitude encoding.
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector
from qiskit.aqua.circuits import StateVectorCircuit
from qiskit.aqua.algorithms import QSVM
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.visualization import circuit_drawer
import numpy as np
import qdata as qd
import torch


def u2Reuploading(nqubits=8, nfeatures=16):
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

# Test feature map with good expressibility and entanglement capability from the work: https://zenodo.org/record/4298781
# Circuit 14:


def get_circuit14(nqubits=4, nfeatures=16, reps=1):
	x = ParameterVector('x', nfeatures)
	qc = QuantumCircuit(nqubits)
	# Ry rotations for the first 4 features
	for irep in range(reps):
	# transform all [0,1] (autoencoder) features to [0,2pi] range
	# maybe better results
		for i in range(4):
			qc.ry(2*np.pi*x[i], i)
		qc.barrier()
		qc.crx(2*np.pi*x[4], 3, 0)
		qc.crx(2*np.pi*x[5], 2, 3)
		qc.crx(2*np.pi*x[6], 1, 2)
		qc.crx(2*np.pi*x[7], 0, 1)
		qc.barrier()
		for i, iq in zip(range(8, 12), range(4)):
			qc.ry(2*np.pi*x[i], iq)
		qc.barrier()
		qc.crx(2*np.pi*x[12], 3, 2)
		qc.crx(2*np.pi*x[13], 0, 3)
		qc.crx(2*np.pi*x[14], 1, 0)
		qc.crx(2*np.pi*x[15], 2, 1)
		qc.barrier()
	return qc
# Circuit 14 but with input data functionality to be used in FeatureMap class


def get_circuitInputs(x, nqubits=4, nfeatures=16, reps=1):
	# transform all [0,1] (autoencoder) features to [0,2pi] range
	qc = QuantumCircuit(nqubits)
	# Ry rotations for the first 4 features
	for irep in range(reps):
		for i in range(4):
			qc.ry(x[i], i)
		qc.barrier()
		qc.crx(x[4], 3, 0)
		qc.crx(x[5], 2, 3)
		qc.crx(x[6], 1, 2)
		qc.crx(x[7], 0, 1)
		qc.barrier()
		for i, iq in zip(range(8, 12), range(4)):
			qc.ry(x[i], iq)
		qc.barrier()
		qc.crx(x[12], 3, 2)
		qc.crx(x[13], 0, 3)
		qc.crx(x[14], 1, 0)
		qc.crx(x[15], 2, 1)
		qc.barrier()
	return qc


class customFeatureMap(RawFeatureVector):
	def construct_circuit(self, x, qr=None, inverse=False):
		"""
		Construct the second order expansion based on given data.

		Args:
		    x (numpy.ndarray): 1-D to-be-encoded data.
		    qr (QuantumRegister): the QuantumRegister object for the circuit, if None,
		                          generate new registers with name q.
		    inverse (bool): inverse
		Returns:
		    QuantumCircuit: a quantum circuit transform data x.
		Raises:
		    TypeError: invalid input
		    ValueError: invalid input
		"""
		if len(x) != self._feature_dimension:
		    raise ValueError("Unexpected feature vector dimension.")

		state_vector = np.pad(x, (0, (1 << self.num_qubits) - len(x)), 'constant')

		svc = StateVectorCircuit(state_vector)

		# Add additional gates after amplitude encoding circuit
		qc = svc.construct_circuit(register=qr)
		# for iqubit in range(qc.num_qubits):
		#	qc.h(iqubit)
		# qc.cx(0,1)
		# qc.cx(1,2)
		# qc.cx(2,3)
		qc += get_circuitInputs(x, nqubits=4, nfeatures=16, reps=1)
		# qc += svc.construct_circuit(register=qr)
		return qc


if __name__ == '__main__':
	qdata = qd.qdata(encoder='pt')
	train = qdata.train
	# train *=2*np.pi#Try to test what changes when data range = [0,1]->[0,2pi]
	# labels = qdata.train_nlabels
	# feature_dim = 4 #TODO should it be named feature_dim or qubits?
	# feature_map = RawFeatureVector(2**feature_dim)#TODO:Use stateVectorCircuit check if can input that in qsvm class
	pls = customFeatureMap(2**4)
	circ = pls.construct_circuit(train[0])
	# print(circ)  
	# print('\nCircuit 14:')
	# print(get_circuit14(nqubits=4,nfeatures=16,reps=1))
	# print('\n u2Reuploading 8 qubits:')
	print(u2Reuploading(nqubits=8,nfeatures=16))
