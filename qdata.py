import numpy as np
from aePyTorch.splitDatasets import splitDatasets

class qdata:

	#Load the test dataset of the autoencoder and split it to training,validation and testing datasets for the qml
	infiles = ('input_ae/trainingTestingDataSig.npy','input_ae/trainingTestingDataBkg.npy')
	trainSigAE, trainBkgAE, validSigAE, validBkgAE, testSigAE,testBkgAE = splitDatasets(infiles,separate=True, not_all = False)
	
	ntot_test = int(testSigAE.shape[0])
	ntot_train = int(trainSigAE.shape[0])
	ntot_valid = int(validSigAE.shape[0])


	def __init__(self, train_p = 0.001, valid_p = 0.004, test_p = 0.004, proportion = True):
		if proportion:
			ntrain = int(self.ntot_train*train_p)
			nvalid = int(self.ntot_valid*valid_p)
			ntest = int(self.ntot_test*test_p)
		else:
			ntrain = train_p
			nvalid = valid_p
			ntest = test.p

		self.ntrain = ntrain
		self.nvalid = nvalid
		self.ntest = ntest

		#print(f'Loaded data for Quantum classifier: ntrain = {ntrain}, nvalid = {nvalid}, ntest = {ntest} ')

		self.train = np.vstack((self.trainSigAE[:ntrain], self.trainBkgAE[:ntrain]))
		self.train_labels = np.array(['s'] * ntrain + ['b'] * ntrain)
		self.train_dict = {'s': self.trainSigAE[:ntrain], 'b': self.trainBkgAE[:ntrain]}

		self.validation = np.vstack((self.validSigAE[:nvalid],self.validBkgAE[:nvalid]))
		self.validation_labels = np.array(['s'] * nvalid + ['b'] * nvalid)
		self.validation_dict = {'s': self.validSigAE[:nvalid], 'b': self.validBkgAE[:nvalid]}

		self.test = np.vstack((self.testSigAE[:ntest],self.testBkgAE[:ntest]))
		self.test_labels = np.array(['s'] * ntest + ['b'] * ntest)
		self.test_dict = {'s': self.testSigAE[:ntest], 'b': self.testBkgAE[:ntest]}

		print(f'xcheck: train/validation/test shapes: {self.train.shape}/{self.validation.shape}/{self.test.shape}')
