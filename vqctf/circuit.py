import pennylane as qml
import numpy as np

from vqctf.fmaps import *
from vqctf.vforms import *

from pennylane.templates import AmplitudeEmbedding


state_0 = [[1], [0]]
y = state_0 * np.conj(state_0).T

def get_layer(spec, nqubits, inputs, theta):
	name = spec[0]
	nfrom = int(spec[1])
	nto = int(spec[2])

	if (name == "zzfm"):
		zzfm(nqubits, inputs[nfrom:nto])
	elif (name == "zzfm2"):
		zzfm(nqubits, inputs[nfrom:nto], scaled = True)
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
	nqubits = spec[0]
	dev = qml.device("default.qubit", wires=nqubits)
	
	@qml.qnode(dev, interface="tf")
	def qcircuit(inputs, theta):
		for l in range(1,len(spec)):
			get_layer(spec[l], nqubits, inputs, theta)
			
		return qml.expval(qml.Hermitian(y, wires=[0]))
	
	return qcircuit





