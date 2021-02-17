from qiskit.aqua.algorithms import QSVM
import numpy as np
from feature_map_testing import customFeatureMap,get_circuit14,u2Reuploading
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from aePyTorch.splitDatasets import splitDatasets
from os import listdir
import time, sys, argparse
import qdata as qd
from encodePT import encode
from datetime import datetime

#Test qsvm models accross different subsets of validation/testing data
start_time = time.time()

infiles = ('input_ae/trainingTestingDataSig7.2e5.npy','input_ae/trainingTestingDataBkg7.2e5.npy')
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type = str,required = True,help='path to saved QSVM model')
parser.add_argument('--k', type = int,default=5,help='number of validation folds for testing')
args = parser.parse_args()

qdata = qd.qdata(encoder='pt',valid_p=0.004)
defaultlayers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size

print('\nLoaded Model: ',args.model)
#feature_map = u2Reuploading(nqubits=8,nfeatures=16)
nqubits = 4
feature_map = RawFeatureVector(2**nqubits)
#feature_map = customFeatureMap(2**nqubits)
backend = Aer.get_backend('statevector_simulator')
#backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend)
print('\nUsing feature map:\n', feature_map.construct_circuit(encode(qdata.train,savedModel,defaultlayers)[0]))

qsvm = QSVM(feature_map, quantum_instance = quantum_instance)

#TODO: Load more than one QSVM models and check their parameters?
folder = 'qsvm/'
model_paths = [folder+ifile for ifile in listdir('qsvm/') if '.npz' in ifile]#Create a list of models saved in .npz files

qsvm.load_model(args.model)

validation_folds = qdata.get_kfold_validation(k=args.k,splits_total = 250)
#[0,1]->[0,2pi]
#validation_folds *=2*np.pi

labels = qdata.validation_labels#TODO: change this automatically if splits_total is changed in get_kfold_validation()
print(labels.shape)
validation_labels = np.where(labels == 's',1,0)#Always the same, thus outside for-loop
accuracies = []
for i,fold in enumerate(validation_folds):
	validation = encode(fold,savedModel,defaultlayers)	
	print(f'fold {i}, shape = {validation.shape}')	
	acc_validation = qsvm.test(validation,validation_labels)
	accuracies.append(acc_validation)
	print(f'Validation Accuracy = {acc_validation}')

accuracies = np.array(accuracies)


print(f'Validation Accuracy mean = {np.mean(accuracies):.4f}, std = {np.std(accuracies):.4f},for k={args.k} validation datasets')
end_time = time.time()
print(f'{args.k}-folds Total Runtime = {end_time-start_time}')

original_stdout = sys.stdout
with open(args.model[:-4]+'.txt','a+') as f:
	sys.stdout = f # Change the standard output to the file we created.
	print(f'Validation Accuracy mean = {np.mean(accuracies):.4f}, std = {np.std(accuracies):.4f},for k={args.k} validation datasets')
	end_time = time.time()
	print(f'{args.k}-folds Total Runtime = {end_time-start_time}/60:.4f min.')
	print('-------------------------------------------\n')
	sys.stdout = original_stdout # Reset the standard output to its original value
