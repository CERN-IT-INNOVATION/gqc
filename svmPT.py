import numpy as np
from aePyTorch.splitDatasets import splitDatasets
import argparse, time, sys
from datetime import datetime
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from encodePT import encode
import qdata as qd 

start_time = time.time()

#TODO Have accessible Ntrain variable
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]
#defaultKernels = ['linear', 'poly', 'rbf', 'sigmoid']
defaultKernels = ['linear', 'rbf']

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--kernel', default=defaultKernels, nargs='+', help='Choose the kernel for sklearn.svm, or a list to train on all')
parser.add_argument('--noEncoding', action = 'store_true', help='If activated, the dataset will be used directly instead of the latent space')

args = parser.parse_args()
(savedModel,layers,kernels) = (args.model,args.layers,args.kernel)

#Data loading:
qdata = qd.qdata(encoder = '' if args.noEncoding else 'pt')
#layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size

#Top 16 features wrt individual AUCiness:
cols = [32, 24, 40, 16, 8, 0, 48, 3, 51, 19, 11, 43, 27, 35, 64, 47]
train = qdata.train[:,cols]

train_labels = qdata.train_labels
validation_labels = qdata.validation_labels
validation_datasets = qdata.get_kfold_validation()
validation_datasets = validation_datasets[:,:,cols]

print(f'Training samples for SVM: {len(train)}.')
print(f'Validation samples for SVM: {validation_datasets.shape[1]}.')#check if correct
#Save results in log:
'''
with open('SVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f # Change the standard output to the file we created.
	print(f'\n---------------------{datetime.now()}----------------------')
	print('Autoencoder model:', savedModel)
	print(f'ntrain = {len(train)}, nvalidation = {len(validation)}.')
	sys.stdout = original_stdout # Reset the standard output to its original value
#Training:
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]

'''

for ikernel in kernels:
#svc=svm.SVC()
#cls = GridSearchCV(svc, param_grid)
#cls.fit(train,train_labels)#Here tested CV with the training dataset
	print('\n')
	print('Kernel: '+ikernel)
	cls = svm.SVC(kernel=ikernel)
	cls.fit(train, train_labels)
	y_scores = np.array([cls.decision_function(fold) for fold in validation_datasets])
	#np.save('qsvm/svm_'+ikernel+'_yscoreList'+('INPUT' if args.noEncoding else '')+'.npy',y_scores)
	np.save('qsvm/svm_'+ikernel+'_yscoreList'+('16AUC' if args.noEncoding else '')+'.npy',y_scores)
	
	print(f'Number of support vectors: {cls.support_vectors_.shape}, #SVs/n_training={len(cls.support_vectors_)/(qdata.ntrain*2):.3f}')
	
#TODO:Maybe delete the accuracies part if we are to use AUROC's from now on...
	'''
	res_train = np.array(cls.predict(train))
	acc_train = sum(res_train == train_labels)/len(res_train)
	
	print(f"Training Accuracy: {acc_train}")
	
	accuracies = []#TODO: make it more pythonic/efficient
	print(f"\nValidation shape {validation_datasets[0].shape}:")
	for fold in validation_datasets:
		fold = encode(fold,savedModel,layers)
		res_validation = np.array(cls.predict(fold))
		acc_validation = sum(res_validation == validation_labels)/len(res_validation)#FIXME: if validation shape[0] not default 288 raises error
		#print(f"Validation Accuracy: {acc_validation}")
		accuracies.append(acc_validation)
	accuracies = np.array(accuracies)
	print(f'Validation Accuracy mean = {np.mean(accuracies):.4f}, std = {np.std(accuracies):.4f},for k={len(validation_datasets)} validation datasets')
	'''
	end_time = time.time()
##	with open('SVMlog.txt', 'a+') as f:
##		original_stdout = sys.stdout
##		sys.stdout = f # Change the standard output to the file we created.
##		print(f'\nKernel: {cls.kernel}')
	print(f'Execution Time {end_time-start_time:.5f} s or {(end_time-start_time)/60:.5f} min.')
#		print(f'Validation Accuracy: {acc_validation}, Training Accuracy: {acc_train}')
#		sys.stdout = original_stdout # Reset the standard output to its original value
#with open('SVMlog.txt','a+') as f:
#	sys.stdout = f # Change the standard output to the file we created.
#	print('-------------------------------------------\n')
#	sys.stdout = original_stdout # Reset the standard output to its original value

