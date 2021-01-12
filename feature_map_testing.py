from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qiskit.aqua.circuits import StateVectorCircuit
from qiskit.aqua.algorithms import QSVM
from qiskit import QuantumCircuit
import numpy as np
import qdata
from encodePT import encode#, device
import torch

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

            #Add additional gates after amplitude encoding circuit
            qc = svc.construct_circuit(register=qr)
            for iqubit in range(qc.num_qubits):
                qc.h(iqubit)
            qc.cx(0,1)
            qc.cx(1,2)
            qc.cx(2,3)
            return qc


if __name__ == '__main__':
    savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr3e-03SigmoidLatent/"
    layers = [67,64, 52, 44, 32, 24, 16]

    trainTest = torch.Tensor(qdata.train)
    train = encode(trainTest,savedModel,layers)
    labels = qdata.train_labels

    labels = np.where(labels =='s',1,0)

    feature_dim = 4 #TODO should it be named feature_dim or qubits?
    feature_map = RawFeatureVector(2**feature_dim)#TODO:Use stateVectorCircuit check if can input that in qsvm class

    pls = customFeatureMap(2**4)
    circ = pls.construct_circuit(train[-1])
    print(circ)
