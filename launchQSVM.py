#from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from feature_map_testing import customFeatureMap,get_circuit
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
from encodePT import encode#, device
import time, sys, argparse, qdata
from datetime import datetime

start_time = time.time()

savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')

args = parser.parse_args()
(savedModel,layers) = (args.model,args.layers)

layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size

#trainTest = torch.Tensor(qdata.train)
train = encode(qdata.train,savedModel,layers)
labels = qdata.train_labels

train_labels = np.where(labels =='s',1,0)

feature_dim = 4 #TODO should it be named feature_dim or qubits?
#feature_map = RawFeatureVector(2**feature_dim)#TODO:Use stateVectorCircuit check if can input that in qsvm class
feature_map = customFeatureMap(2**feature_dim)
#train *= np.pi #map to [0,pi]#TEST
#feature_map = get_circuit(nqubits=4,nfeatures=16,reps=2)

backend = Aer.get_backend('statevector_simulator')
#backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend)

qsvm = QSVM(feature_map, quantum_instance = quantum_instance,lambda2 = 0.2)#lambda2 Increase soft margin regularization 
#qsvm.train(train,train_labels)

#testTest = torch.Tensor(qdata.test)
test = encode(qdata.test,savedModel,defaultlayers)
labels = qdata.test_labels
test_labels = np.where(labels == 's',1,0)

acc_test = qsvm.test(test,test_labels)
acc_train = qsvm.test(train,train_labels)
print(f'Test Accuracy = {acc_test}')
print(f'Training Accuracy = {acc_train}')
end_time = time.time()

#TODO: Bariers for visualizing better the feature map in the log
#TODO: rename test to validation in log and printing
with open('QSVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f
	print(f'\n---------------------{datetime.now()}----------------------')
	print('Autoencoder model:', savedModel)
	print(f'ntrain = {len(train)}, ntest = {len(test)}, lambda2 = {qsvm.lambda2}')
	print(f'Quantum Instance backend:{quantum_instance.backend}')
	#print(f'Quantum Instance basic info: {quantum_instance}')
	print(f'Execution Time {end_time-start_time} s or {(end_time-start_time)/60} min.')
	print(f'Test Accuracy: {acc_test}')
	print(f'Training Accuracy: {acc_train}')	
	#print(f'Feature Map\n: {get_circuit(nqubits=4,nfeatures=16,reps=2)}')
	print(f'Feature Map:\n\n {feature_map.construct_circuit(train[0])}')
	print('-------------------------------------------\n')
	sys.stdout = original_stdout # Reset the standard output to its original value
#TODO: Add save model
qsvm.save_model('qsvm/model4')
