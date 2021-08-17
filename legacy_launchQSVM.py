
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qsvm.feature_map_circuits import customFeatureMap,get_circuit14,u2Reuploading
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
import time, sys, argparse
import qdata as qd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)#for aqua


seed = 12345

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model_path',type=str,default='/work/vabelis/qml_project/'
		'autoencoder_pytorch/trained_models/vanilla_vasilis',
		help='path to saved model')
parser.add_argument('--name', required = True,help='Model name to be saved')

args = parser.parse_args()


def main():
	start_time = time.time()
	feature_dim=16
	qdata_loader = qd.qdata(data_folder = '../qml_data/input_ae/', norm_name = 'minmax',
		nevents = '7.20e+05', model_path= args.model_path, train_events=200,
		 valid_events=720, test_events=50)

	train_features = qdata_loader.get_latent_space('train')
	train_labels   = qdata_loader.ae_data.train_target
	test_features  = qdata_loader.get_latent_space('test')
	test_labels    = qdata_loader.ae_data.test_target

	feature_map = RawFeatureVector(feature_dim)
	backend = QuantumInstance(Aer.get_backend('statevector_simulator'),
		seed_simulator=seed, seed_transpiler = seed)
	
	qsvm = QSVM(feature_map, quantum_instance = backend,lambda2=0.2)
	qsvm.train(train_features, train_labels)

	# Compute the accuracies for quick probe for overtraining
	test_accuracy = qsvm.test(test_features, test_labels)
	train_accuracy = qsvm.test(train_features,train_labels)
	
	print(f'Test Accuracy = {test_accuracy}')
	print(f'Training Accuracy = {train_accuracy}')

	end_time = time.time()
	runtime = end_time-start_time
	print(f'Total runtime: {runtime:.2f} sec.')
	
	save_model_log(qdata_loader,qsvm,runtime,train_accuracy,test_accuracy)


def save_model_log(qdata_loader,qsvm_model,runtime,train_accuracy,test_accuracy):
	with open('qsvm/QSVMlog_new.txt', 'a+') as f:
		original_stdout = sys.stdout
		sys.stdout = f
		print(f'\n---------------------{datetime.now()}----------------------')
		print('Autoencoder model path:', args.model_path)
		print('Data path:', qdata_loader.ae_data.data_folder)
		print(f'ntrain = {len(qdata_loader.ae_data.train_target)}, '
				f'ntest = {len(qdata_loader.ae_data.test_target)}, '
				f'lambda2 = {qsvm_model.lambda2}')
		print(f'Execution Time {runtime} s or {runtime/60} min.')
		print(f'Test Accuracy: {test_accuracy}, Training Accuracy: {train_accuracy}')
		print('-------------------------------------------\n')
		sys.stdout = original_stdout
	
	qsvm_model.save_model('qsvm/'+args.name)


if __name__ == '__main__':
	main()