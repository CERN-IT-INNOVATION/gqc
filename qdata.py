import numpy as np
from aePyTorch.model import *
from aePyTorch.splitDatasets import *

#Load the test dataset of the autoencoder and split it to training,validation and testing datasets for the qml
infiles = ('input_ae/trainingTestingDataSig.npy','input_ae/trainingTestingDataBkg.npy')
testSigAE,testBkgAE = splitDatasets(infiles,separate=True)

ntot = int(testSigAE.shape[0])
if testSigAE.shape[0] != testBkgAE.shape[0]:
	raise Exception('nSig != nBkg!!! Events should be equal')

ntrain, nvalid, ntest = int(ntot*0.8), int(0.1*ntot), int(0.1*ntot)
print(f'Loaded data for Quantum classifier: ntrain = {ntrain}, nvalid = {nvalid}, ntest = {ntest} ')

train = np.vstack((testSigAE[:ntrain],testBkgAE[:ntrain]))
train_labels = ['s'] * ntrain + ['b'] * ntrain;
validation = np.vstack((testSigAE[ntrain:(ntrain+nvalid)],testBkgAE[ntrain:(ntrain+nvalid)]))
validation_labels = ['s'] * nvalid + ['b'] * nvalid;
test = np.vstack((testSigAE[(ntrain+nvalid):],testBkgAE[(ntrain+nvalid):]))
test_labels = ['s'] * ntest + ['b'] * ntest;


if np.array_equal(validation,test):
	raise Exception('Validation and Testing datasets are the same!')

print(f'xcheck: train/validation/test shapes: {train.shape}/{validation.shape}/{test.shape}')

