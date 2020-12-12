import numpy as np
from aePyTorch.model import *
from aePyTorch.splitDatasets import *

#Load the test dataset of the autoencoder and split it to training,validation and testing datasets for the qml
infiles = ('input_ae/trainingTestingDataSig.npy','input_ae/trainingTestingDataBkg.npy')
trainSigAE, trainBkgAE, validSigAE, validBkgAE, testSigAE,testBkgAE = splitDatasets(infiles,separate=True, not_all = False)

ntot_test = int(testSigAE.shape[0])
ntot_train = int(trainSigAE.shape[0])
ntot_valid = int(validSigAE.shape[0])
if ntot_test != testBkgAE.shape[0] or ntot_train != trainSigAE.shape[0] or ntot_valid != validSigAE.shape[0]:
	raise Exception('nSig != nBkg!!! Events should be equal')

ntrain, nvalid, ntest = int(ntot_test*0.1), int(0.1*ntot_test), int(0.1*ntot_test)
print(f'Loaded data for Quantum classifier: ntrain = {ntrain}, nvalid = {nvalid}, ntest = {ntest} ')

train = np.vstack((trainSigAE[:ntrain],trainBkgAE[:ntrain]))
train_labels = ['s'] * ntrain + ['b'] * ntrain;
train_dict = {'s': trainSigAE[:ntrain], 'b': trainBkgAE[:ntrain]}

validation = np.vstack((validSigAE[:nvalid],validBkgAE[:nvalid]))
validation_labels = ['s'] * nvalid + ['b'] * nvalid;
validation_dict = {'s': validSigAE[:ntrain], 'b': validBkgAE[:ntrain]}


test = np.vstack((testSigAE[:ntest],testBkgAE[:ntest]))
test_labels = ['s'] * ntest + ['b'] * ntest;

if np.array_equal(validation,test):
	raise Exception('Validation and Testing datasets are the same!')

print(f'xcheck: train/validation/test shapes: {train.shape}/{validation.shape}/{test.shape}')

