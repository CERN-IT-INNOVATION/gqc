from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from sklearn.svm import SVC
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np
import time, sys, argparse
from datetime import datetime

import qdata as qd

from qsvm.feature_map_circuits import u2Reuploading

seed = 12345
algorithm_globals.random_seed = seed #ensure same global behaviour

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
	default = '../qml_data/input_ae/', 
	help = "The folder where the data is stored on the system.")
parser.add_argument("--norm", type=str, default='minmax',
    help = "The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str, default='7.20e+05',
    help = "The number of events of the norm file.")
parser.add_argument('--model_path', type=str, 
    default = 'autoencoder_pytorch/trained_models/vanilla_vasilis', 
	help = "The path to the saved model.")
#parser.add_argument('--output_file', required = True,
#	help = 'Model name to be saved.')
'''
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--name', required = True,help='Model name to be saved')
parser.add_argument('--noEncoding',action = 'store_true', help = 'If activated, the dataset will be used directly instead of latent space')

(savedModel,layers) = (args.model,args.layers)
'''

args = parser.parse_args()

def main():
	start_time = time.time()
	
	feature_dim=16
	
	qdata_loader = qd.qdata(args.data_folder, args.norm, args.nevents,
	args.model_path, train_events=6, valid_events=6, test_events=6)

	train_features = qdata_loader.get_latent_space('train')
	train_labels   = qdata_loader.ae_data.train_target
	test_features  = qdata_loader.get_latent_space('test')
	test_labels    = qdata_loader.ae_data.test_target

	feature_map = u2Reuploading(nqubits=8, nfeatures=feature_dim)
	backend = QuantumInstance(Aer.get_backend('aer_simulator_statevector'),
		seed_simulator=seed, seed_transpiler = seed)
	quantum_kernel = QuantumKernel(feature_map = feature_map,
		quantum_instance = backend)
	
	qsvm = SVC(kernel=quantum_kernel.evaluate)
	qsvm.fit(train_features, train_labels)

	print('Train labels:', train_labels)
	print('Train data:', train_features)
	
	test_accuracy = qsvm.score(test_features, test_labels)
	train_accuracy = qsvm.score(train_features,train_labels)

	#Quick probe for overtraining
	print(f'Test Accuracy = {test_accuracy}')
	print(f'Training Accuracy = {train_accuracy}')

	end_time = time.time()
	runtime = end_time-start_time
	print(f'Total runtime: {runtime:.2f} sec.')
	#qsvm.save_model('qsvm/best_model')


if __name__ == '__main__':
	main()

'''

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

'''
