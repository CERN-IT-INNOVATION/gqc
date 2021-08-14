from qiskit import Aer
from qsvm.feature_map_circuits import customFeatureMap,get_circuit14,u2Reuploading
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from sklearn.svm import SVC
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np
from qiskit_machine_learning.datasets import wine#test if this produces "bound" parameters for RawFeatureVector
#from encodePT import encode#, device
import time, sys, argparse
from datetime import datetime
import qdata as qd

from qiskit_machine_learning.datasets import ad_hoc_data

seed = 12345
algorithm_globals.random_seed = seed #ensure same global behaviour (?)


'''
#TODO: dictionary with feature maps <-> nqubits etc like compute_scores.py
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--name', required = True,help='Model name to be saved')
parser.add_argument('--noEncoding',action = 'store_true', help = 'If activated, the dataset will be used directly instead of latent space')

args = parser.parse_args()
(savedModel,layers) = (args.model,args.layers)

#Define the dataset object:
qdata = qd.qdata('' if args.noEncoding else 'pt')
#layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size
'''

def main():
	start_time = time.time()
	feature_dim=4
	training_size = 12
	test_size = 4

	train_features, train_labels, test_features, test_labels = \
	wine(training_size=training_size, test_size=test_size, n=feature_dim)
	
	feature_map = RawFeatureVector(feature_dim)
	backend = QuantumInstance(Aer.get_backend('aer_simulator_statevector'),seed_simulator=seed,seed_transpiler = seed)
	quantum_kernel = QuantumKernel(feature_map = feature_map, quantum_instance = backend)
	
	qsvm = SVC(kernel = quantum_kernel.evaluate)
	qsvm.fit(train_features,train_labels)
	print('Train labels:', train_labels)
	print('Train data:', train_features)
	test_accuracy = qsvm.score(test_features, test_labels)
	
	end_time = time.time()
	runtime = end_time-start_time
	print(f'Total runtime: {runtime:.2f} sec.')

if __name__ == '__main__':
	main()

'''
#trainTest = torch.Tensor(qdata.train)
train = qdata.train
#These are the 16 features with best individual AUCiness of the input space:
cols = [32, 24, 40, 16, 8, 0, 48, 3, 51, 19, 11, 43, 27, 35, 64, 47]
train = train[:,cols]
#train = encode(qdata.train,savedModel,layers)


Testing
test = qdata.test
test = test[:,cols]
#test = np.delete(test,[34,42,50],1)
#test = encode(qdata.test,savedModel,defaultlayers)

acc_test = qsvm.test(test,qdata.test_nlabels)
acc_train = qsvm.test(train,qdata.train_nlabels)
print(f'Test Accuracy = {acc_test}')
print(f'Training Accuracy = {acc_train}')

#TODO: rename test to validation in log and printing
with open('QSVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f
	print(f'\n---------------------{datetime.now()}----------------------')
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
'''
