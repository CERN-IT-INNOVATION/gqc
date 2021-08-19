
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
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
parser.add_argument('--model_path', type=str, required=True,
    help="The path to the saved model.")
parser.add_argument('--output_file', required = True,
	help='Model name to be saved.')

args = parser.parse_args()


def main():
	start_time  = time.time()
	feature_dim = 16
	qdata_loader = qd.qdata(args.data_folder, args.norm, args.nevents,
	 	args.model_path, train_events=200, valid_events=50, test_events=720)

	train_features = qdata_loader.get_latent_space('train')
	train_labels   = qdata_loader.ae_data.train_target
	valid_features = qdata_loader.get_latent_space('valid')
	valid_labels   = qdata_loader.ae_data.valid_target

	feature_map = RawFeatureVector(feature_dim)
	backend = QuantumInstance(Aer.get_backend('statevector_simulator'),
		seed_simulator=seed, seed_transpiler=seed)
	
	qsvm = QSVM(feature_map, quantum_instance=backend, lambda2=0.3)
	qsvm.train(train_features, train_labels)

	# Compute the accuracies for quick probe for overtraining
	valid_accuracy = qsvm.test(valid_features, valid_labels)
	train_accuracy = qsvm.test(train_features, train_labels)
	
	print(f'Test Accuracy = {valid_accuracy}')
	print(f'Training Accuracy = {train_accuracy}')

	end_time = time.time()
	runtime = end_time-start_time
	print(f'Total runtime: {runtime:.2f} sec.')
	
	save_model_log(qdata_loader, qsvm, runtime, train_accuracy, valid_accuracy)


def save_model_log(qdata_loader, qsvm_model, runtime, train_accuracy,
	valid_accuracy):
	print(f'\n---------------------{datetime.now()}----------------------')
	print('Autoencoder model path:', args.model_path)
	print('Data path:', qdata_loader.ae_data.data_folder)
	print(f'ntrain = {len(qdata_loader.ae_data.train_target)}, '
			f'ntest = {len(qdata_loader.ae_data.valid_target)}, '
			f'lambda2 = {qsvm_model.lambda2}')
	print(f'Execution Time {runtime} s or {runtime/60} min.')
	print(f'Test Accuracy: {valid_accuracy}, '
	      f'Training Accuracy: {train_accuracy}')
	print('-------------------------------------------\n')
	
	qsvm_model.save_model('qsvm_output/' + args.output_file)

if __name__ == '__main__':
	main()
