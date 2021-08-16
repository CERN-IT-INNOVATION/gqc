
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qsvm.feature_map_circuits import customFeatureMap,get_circuit14,u2Reuploading
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
#from qiskit.utils import algorithm_globals
import numpy as np
import time, sys, argparse
import qdata as qd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)#for aqua


seed = 12345
#algorithm_globals.random_seed = seed
'''
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--name', required = True,help='Model name to be saved')
parser.add_argument('--noEncoding',action = 'store_true', help = 'If activated, the dataset will be used directly instead of latent space')

args = parser.parse_args()
(savedModel,layers) = (args.model,args.layers)
'''

def main():
	start_time = time.time()
	feature_dim=16
	qdata_loader = qd.qdata(data_folder = '../qml_data/input_ae/', norm_name = 'minmax',
		nevents = '7.20e+05', model_path= '/work/vabelis/qml_project/autoencoder_pytorch/'
		'trained_models/vanilla_best', train_events=20, valid_events=10, test_events=10)

	train_features = qdata_loader.get_latent_space('train')
	train_labels   = qdata_loader.ae_data.train_target
	test_features  = qdata_loader.get_latent_space('test')
	test_labels    = qdata_loader.ae_data.test_target

	feature_map = RawFeatureVector(feature_dim)
	backend = QuantumInstance(Aer.get_backend('statevector_simulator'),
		seed_simulator=seed, seed_transpiler = seed)
	
	qsvm = QSVM(feature_map, quantum_instance = backend,lambda2 = 0.2)
	qsvm.train(train_features, train_labels)

	print(train_features.shape)
	print('Train labels:', train_labels)
	print('Train data:', train_features)
	test_accuracy = qsvm.test(test_features, test_labels)
	train_accuracy = qsvm.test(train_features,test_labels)
	
	print(f'Test Accuracy = {test_accuracy}')
	print(f'Training Accuracy = {train_accuracy}')

	end_time = time.time()
	runtime = end_time-start_time
	print(f'Total runtime: {runtime:.2f} sec.')
	#qsvm.save_model('qsvm/best_model')


if __name__ == '__main__':
	main()

'''

acc_test = qsvm.test(test,qdata.test_nlabels)
acc_train = qsvm.test(train,qdata.train_nlabels)

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
'''