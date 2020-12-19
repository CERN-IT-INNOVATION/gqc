from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
import qdata
from encodePT import encode#, device
import torch, time, sys, argparse

start_time = time.time()

savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr3e-03SigmoidLatent/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')

args = parser.parse_args()
(savedModel,layers) = (args.model,args.layers)

layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size

trainTest = torch.Tensor(qdata.train)
train = encode(trainTest,savedModel,layers)
labels = qdata.train_labels

labels = np.where(labels =='s',1,0)

feature_dim = 4 #TODO should it be named feature_dim or qubits?
feature_map = RawFeatureVector(2**feature_dim)#TODO:Use stateVectorCircuit check if can input that in qsvm class

#backend = Aer.get_backend('statevector_simulator')
backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend)
qsvm = QSVM(feature_map, quantum_instance = quantum_instance)
qsvm.train(train,labels)

testTest = torch.Tensor(qdata.test)
test = encode(testTest,savedModel,defaultlayers)
labels = qdata.test_labels

test_labels = np.where(labels == 's',1,0)
acc = qsvm.test(test,test_labels)
print(f'Accuracy = {acc}')

end_time = time.time()

with open('QSVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f
	print('\n-------------------------------------------')
	print('Autoencoder model:', savedModel)
	print(f'ntrain = {len(train)}, ntest = blah')
	print(f'Quantum Instance: {quantum_instance}')
	print(f'\nExecution Time {end_time-start_time} s or {(end_time-start_time)/60} min.')
	print(f'Accuracy: {acc}')
	print('-------------------------------------------\n')
	sys.stdout = original_stdout # Reset the standard output to its original value
