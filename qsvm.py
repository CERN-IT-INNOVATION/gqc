from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
import qdata
from aePyTorch.encode import encode#, device
import torch

feature_dim = 4 #TODO should it be named feature_dim or qubits?
feature_map = RawFeatureVector(2**feature_dim)
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr3e-03SigmoidLatent/"#TODO: argparse
defaultlayers = [67,64, 52, 44, 32, 24, 16]

trainTest = torch.Tensor(qdata.train)
train = encode(trainTest,savedModel,defaultlayers)
labels = np.array(qdata.train_labels)

labels = np.where(labels =='s',1,0)

backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)
qsvm = QSVM(feature_map, quantum_instance = quantum_instance)

qsvm.train(train,labels)

testTest = torch.Tensor(qdata.test)
test = encode(testTest,savedModel,defaultlayers)
labels = np.array(qdata.test_labels)

test_labels = np.where(labels == 's',1,0)
print(qsvm.test(test,test_labels))