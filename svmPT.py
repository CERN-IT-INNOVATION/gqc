import numpy as np
import torch
from aePyTorch.splitDatasets import splitDatasets
import argparse, time, sys
from sklearn import svm
from encodePT import encode
import qdata 

start_time = time.time()

#TODO Have accessible Ntrain variable
infiles = ('../input_ae/trainingTestingDataSig.npy','../input_ae/trainingTestingDataBkg.npy')#Not needed if qdata is used
savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr3e-03SigmoidLatent/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--input", type=str, default=infiles, nargs =2, help="path to datasets")#Not needed if qdata is used
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')

#TODO: choose qdata or splitDatasets
parser.add_argument('--qdata', action= 'store_true', help='Activates the use of qdata instead of splitDatasets')

args = parser.parse_args()
infiles = args.input
(savedModel,layers) = (args.model,args.layers)


if args.qdata:
	#qdata samples for fair benchmark with quantum classifiers:
	print('Using  qdata')
	layers.insert(0,qdata.train.shape[1]) #insert number of input nodes = feature_size
	train = encode(qdata.train,savedModel,layers)
	train_labels = qdata.train_labels
	
	test = encode(qdata.test,savedModel,layers)
	test_labels = qdata.test_labels
else:
	#Load samples from splitDatasets
	print('Using ',args.input)
	_,validDataset,testDataset,_,validLabels, testLabels = splitDatasets(infiles, labels=True)
	layers.insert(0,testDataset.shape[1])
	
	train = encode(validDataset,savedModel,layers)
	labels = np.array(validLabels)
	
	test = encode(testDataset,savedModel,layers)
	testlabels = np.array(testLabels)


print(f'Training samples for SVM: {len(train)}.')
print(f'Testing samples for SVM: {len(test)}.')

cls = svm.SVC()
cls.fit(train, train_labels)

res = np.array(cls.predict(test))

acc = sum(res == test_labels) / len(res)
print(f"Accuracy: {acc} ")

end_time = time.time()
with open('SVMlog.txt', 'a+') as f:
	original_stdout = sys.stdout
	sys.stdout = f # Change the standard output to the file we created.
	print('\n-------------------------------------------')
	print('Autoencoder model:', savedModel)
	print(f'ntrain = {len(train)}, ntest = {len(test)}')
	print(f'\nExecution Time {end_time-start_time} s or {(end_time-start_time)/60} min.')
	print(f'Accuracy: {acc}')
	print('-------------------------------------------\n')
	sys.stdout = original_stdout # Reset the standard output to its original value
