from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from feature_map_circuits import customFeatureMap,get_circuit14,u2Reuploading
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
from encodePT import encode#, device
import time, sys, argparse
import qdata as qd
from datetime import datetime

start_time = time.time()

#TODO: dictionary with feature maps <-> nqubits etc like compute_scores.py
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--name', required = True,help='Model name to be saved')
parser.add_argument('--shift_vars', action = 'store_true', help = 'Shift input features to different range')
parser.add_argument('--noEncoding',action = 'store_true', help = 'If activated, the dataset will be used directly instead of latent space')

args = parser.parse_args()
(savedModel,layers) = (args.model,args.layers)

if args.shift_vars:
	print('Shift input features')
else:
	print('No shift of input features')

#Define the dataset object:
qdata = qd.qdata('' if args.noEncoding else 'pt')
#layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size

#trainTest = torch.Tensor(qdata.train)
train = qdata.train
#These are the 16 features with best individual AUCiness of the input space:
cols = [32, 24, 40, 16, 8, 0, 48, 3, 51, 19, 11, 43, 27, 35, 64, 47]
train = train[:,cols]
#train = encode(qdata.train,savedModel,layers)
#train *= 2*np.pi#map [0,1]->[0,2pi] maybe better performance?

n_qubits = 4
#FIXME: put 64 features for training 
#train = np.delete(train,[34,42,50],1)#removing jet7,jet6 and jet5 phi's
#feature_map = customFeatureMap(2**n_qubits)
feature_map = RawFeatureVector(2**n_qubits)
#feature_map = get_circuit14(nqubits=4,nfeatures=16,reps=1)
#feature_map = u2Reuploading(nqubits = 8, nfeatures = 16)
#print(u2Reuploading(nqubits = 8, nfeatures = 16))
#print(f'Feature Map:\n\n {feature_map.construct_circuit(train[0])}')

backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)

qsvm = QSVM(feature_map, quantum_instance = quantum_instance,lambda2 = 0.2)#lambda2 Increase soft margin regularization 
qsvm.train(train,qdata.train_nlabels)

#testTest = torch.Tensor(qdata.test)
test = qdata.test
test = test[:,cols]
#test = np.delete(test,[34,42,50],1)
#test = encode(qdata.test,savedModel,defaultlayers)
#test *=2*np.pi#better performance due to higher expressibility?

acc_test = qsvm.test(test,qdata.test_nlabels)
acc_train = qsvm.test(train,qdata.train_nlabels)
print(f'Test Accuracy = {acc_test}')
print(f'Training Accuracy = {acc_train}')
end_time = time.time()

#TODO: rename test to validation in log and printing
with open('QSVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f
	print(f'\n---------------------{datetime.now()}----------------------')
	if args.shift_vars:
		print(f'With shift of features from [0,1] to [0,2pi]')
	print('Autoencoder model:', savedModel)
	print(f'ntrain = {len(train)}, ntest = {len(test)}, lambda2 = {qsvm.lambda2}')
	print(f'Quantum Instance backend:{quantum_instance.backend}')
	print(f'Execution Time {end_time-start_time} s or {(end_time-start_time)/60} min.')
	print(f'Test Accuracy: {acc_test}, Training Accuracy: {acc_train}')
	#print(f'Feature Map\n: u2Reuploading(nqubits = 8, nfeatures = 16)')
	#print(f'Feature Map:\n\n {feature_map.construct_circuit(train[0])}')
	print('-------------------------------------------\n')
	sys.stdout = original_stdout # Reset the standard output to its original value
qsvm.save_model('qsvm/'+args.name)
