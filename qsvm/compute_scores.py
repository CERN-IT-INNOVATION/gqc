from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
#Custom feature maps:
from feature_map_testing import customFeatureMap,get_circuit14,u2Reuploading
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
import qdata as qd
import time,argparse

#TODO: merge with test.py & have uncertainties on curves and AUC
start_time = time.time()

feature_map_dict = {'amp_enc_only':RawFeatureVector(2**4),'amp_enc_only6':RawFeatureVector(2**6),'u2Reuploading':u2Reuploading(nqubits = 8, nfeatures=16)}

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type = str,required = True,help='Path to saved model excluding .npz')
parser.add_argument('--feature_map',required = True, help = f'Choose from available feature maps: {feature_map_dict.keys()}')
parser.add_argument('--noEncoding',action = 'store_true', help = 'If activated, the dataset will be used directly instead of latent space')
args = parser.parse_args()

qdata = qd.qdata('' if args.noEncoding else 'pt')

backend = Aer.get_backend('statevector_simulator')
qi = QuantumInstance(backend)
feature_map = feature_map_dict[args.feature_map]
qsvm = QSVM(feature_map,quantum_instance = qi)

qsvm.load_model(args.model+'.npz')

validation_folds = qdata.get_kfold_validation() [5,720,64]
#Best input features based on individual AUC values: (test vs autoencoder)
cols = [32, 24, 40, 16, 8, 0, 48, 3, 51, 19, 11, 43, 27, 35, 64, 47]
validation_folds = validation_folds[:,:,cols]
#validation_folds = np.delete(validation_folds,[34,42,50],axis = 2)

y_scores = np.array([qsvm.instance.get_predicted_confidence(fold) for fold in validation_folds])
np.save(args.model+'_yscoreList.npy',y_scores)

end_time = time.time()
print(f'Total Runtime = {(end_time-start_time)/60:.4f} min.')
