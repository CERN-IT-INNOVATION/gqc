import numpy as np
from aePyTorch.splitDatasets import splitDatasets
import argparse, time, sys
from datetime import datetime
from sklearn import svm
from encodePT import encode
import qdata 

start_time = time.time()

#TODO Have accessible Ntrain variable
infiles = ('./input_ae/trainingTestingDataSig7.2e5.npy','./input_ae/trainingTestingDataBkg7.2e5.npy')#Not needed if qdata is used
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
defaultlayers = [64, 52, 44, 32, 24, 16]
defaultKernels = ['linear', 'poly', 'rbf', 'sigmoid']

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--input", type=str, default=infiles, nargs =2, help="path to datasets")#Not needed if qdata is used
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--kernel', default=defaultKernels, nargs='+', help='Choose the kernel for sklearn.svm, or a list to train on all')
parser.add_argument('--qdata', action= 'store_true', help='Activates the use of qdata instead of splitDatasets')
parser.add_argument('--noEncoding', action = 'store_true', help='If activated, the dataset will be used directly instead of the latent space')

args = parser.parse_args()
infiles = args.input
(savedModel,layers,kernels) = (args.model,args.layers,args.kernel)

#Data loading:
if args.qdata:
	#qdata samples for fair benchmark with quantum classifiers:
	print('Using  qdata')
	layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size
	if args.noEncoding:
		#On unencoded data:
		train = qdata.train
		test = qdata.test
	else:
		train = encode(qdata.train,savedModel,layers)
		test = encode(qdata.test,savedModel,layers)
	train_labels = qdata.train_labels
	test_labels = qdata.test_labels

else:
	#Load samples from splitDatasets
	print('Using ',args.input)
	_,validDataset,testDataset,_,validLabels, testLabels = splitDatasets(infiles, labels=True)
	layers.insert(0,testDataset.shape[1])
	
	if args.noEncoding:
	#On unencoded data:
		train = validDataset
		test = testDataset
	else:
		train = encode(validDataset,savedModel,layers)
		test = encode(testDataset,savedModel,layers)
	
	train_labels = np.array(validLabels)
	test_labels = np.array(testLabels)


print(f'Training samples for SVM: {len(train)}.')
print(f'Testing samples for SVM: {len(test)}.')

#Save results in log
with open('SVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f # Change the standard output to the file we created.
	print(f'\n---------------------{datetime.now()}----------------------')
	print('Autoencoder model:', savedModel)
	print(f'ntrain = {len(train)}, ntest = {len(test)}.')
	sys.stdout = original_stdout # Reset the standard output to its original value

#Training:
for ikernel in kernels:
	print('Kernel: '+ikernel)
	cls = svm.SVC(kernel=ikernel)
	cls.fit(train, train_labels)
	
	res_test = np.array(cls.predict(test))
	res_train = np.array(cls.predict(train))
	
	acc_test = sum(res_test == test_labels)/len(res_test)
	acc_train = sum(res_train == train_labels)/len(res_train)
	print(f"Test Accuracy: {acc_test} ")
	print(f"Training Accuracy: {acc_train} ")
	
	end_time = time.time()
	with open('SVMlog.txt', 'a+') as f:
		original_stdout = sys.stdout
		sys.stdout = f # Change the standard output to the file we created.
		print(f'\nKernel: {cls.kernel}')
		print(f'Execution Time {end_time-start_time} s or {(end_time-start_time)/60} min.')
		print(f'Test Accuracy: {acc_test}, Training Accuracy: {acc_train}')
		sys.stdout = original_stdout # Reset the standard output to its original value

with open('SVMlog.txt','a+') as f:
	sys.stdout = f # Change the standard output to the file we created.
	print('-------------------------------------------\n')
	sys.stdout = original_stdout # Reset the standard output to its original value
	
