import pennylane as qml
import numpy as np

from vqctf.fmaps import *
from vqctf.vforms import *

from pennylane.templates import AmplitudeEmbedding
from pennylane.templates import AngleEmbedding



state_0 = [[1], [0]]
state_all = [[1]], 
y = state_0 * np.conj(state_0).T

def get_layer(spec, nqubits, inputs, theta):
	name = spec[0]
	nfrom = int(spec[1])
	nto = int(spec[2])

	if (name == "zzfm"):
		zzfm(nqubits, inputs[nfrom:nto])
	elif (name == "zzfm2"):
		zzfm(nqubits, inputs[nfrom:nto], scaled = True)
	elif (name == "angle"):
		AngleEmbedding(features = inputs[nfrom:nto], wires = range(nqubits))
	elif (name == "angle2"):
		AngleEmbedding(features = np.pi * inputs[nfrom:nto], wires = range(nqubits))
	elif (name == "ae"):
		AmplitudeEmbedding(features = inputs[nfrom:nto], wires = range(nqubits), normalize = True)
	elif (name == "2local"):
		twolocal(nqubits, theta[nfrom:nto], reps = int(spec[3]), entanglement = spec[4])
	elif (name == "tree"):
		treevf(nqubits, theta[nfrom:nto],reps = int(spec[3]))
	elif (name == "step"):
		stepc(nqubits, theta[nfrom:nto],reps = int(spec[3]))
	else:
		raise Exception("Unknown template!")


def get_circuit(spec):
	nqubits = 0
	measure = "1"
	if (isinstance(spec[0], list)):
		nqubits = spec[0][0]
		measure = spec[0][1]
	else:
		nqubits = spec[0]

	dev = qml.device("default.qubit", wires=nqubits)
	
	@qml.qnode(dev, interface="tf")
	def qcircuit(inputs, theta):
		for l in range(1,len(spec)):
			get_layer(spec[l], nqubits, inputs, theta)
		
		if (measure == "all3"):
			return qml.expval(qml.Hermitian(y, wires=[0]) @ qml.Hermitian(y,wires = [1]) @ qml.Hermitian(y,wires=[2]))
		if (measure == "all4"):
			return qml.expval(qml.Hermitian(y, wires=[0]) @ qml.Hermitian(y,wires = [1]) @ qml.Hermitian(y,wires=[2]) @ qml.Hermitian(y,wires = [3]))
		elif (measure == "first"):
			return qml.expval(qml.Hermitian(y, wires = [0]))
		else:
			raise Expception("Undefined measurement")

	
	return qcircuit





