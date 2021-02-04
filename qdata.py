import numpy as np
from aePyTorch.splitDatasets import splitDatasets
from aeTF.encode import encode

class qdata:

	#Load the test dataset of the autoencoder and split it to training,validation and testing datasets for the qml
	#infiles = ('input_ae/trainingTestingDataSig.npy','input_ae/trainingTestingDataBkg.npy')
	infiles = ('input_ae/trainingTestingDataSig7.2e5.npy','input_ae/trainingTestingDataBkg7.2e5.npy')
	trainSigAE, trainBkgAE, validSigAE, validBkgAE, testSigAE,testBkgAE = splitDatasets(infiles,separate=True, not_all = False)
	
	ntot_test = int(testSigAE.shape[0])
	ntot_train = int(trainSigAE.shape[0])
	ntot_valid = int(validSigAE.shape[0])

#TODO: Add encode in constructor by default and make it work for TF and PT.
	def __init__(self, encoder, train_p = 0.0005, valid_p = 0.002, test_p = 0.002, proportion = True):
		if encoder == "tf":
			print('Using tf for autoencoder model')
			self.trainSigAE = encode(self.trainSigAE)
			self.trainBkgAE = encode(self.trainBkgAE)
			self.validSigAE = encode(self.validSigAE)
			self.validBkgAE = encode(self.validBkgAE)
			self.testSigAE = encode(self.testSigAE)
			self.testBkgAE = encode(self.testBkgAE)
		elif encoder == "pt":
			print('Using pt for autoencoder model to encode the data')
			# TODO Implement encoding
		elif encoder == "":
			print("Using unencoded data");
		else:
			raise Exception('Unknown encoder')


		if proportion:
			ntrain = int(self.ntot_train*train_p)
			nvalid = int(self.ntot_valid*valid_p)
			ntest = int(self.ntot_test*test_p)
		else:
			ntrain = train_p
			nvalid = valid_p
			ntest = test_p

		self.ntrain = ntrain
		self.nvalid = nvalid
		self.ntest = ntest

		#print(f'Loaded data for Quantum classifier: ntrain = {ntrain}, nvalid = {nvalid}, ntest = {ntest} ')

		self.train = np.vstack((self.trainSigAE[:ntrain], self.trainBkgAE[:ntrain]))
		self.train_labels = np.array(['s'] * ntrain + ['b'] * ntrain)
		self.train_nlabels = np.array([1] * ntrain + [0] * ntrain)
		self.train_dict = {'s': self.trainSigAE[:ntrain], 'b': self.trainBkgAE[:ntrain]}

		self.validation = np.vstack((self.validSigAE[:nvalid],self.validBkgAE[:nvalid]))
		self.validation_labels = np.array(['s'] * nvalid + ['b'] * nvalid)
		self.validation_nlabels = np.array([1] * nvalid + [0] * nvalid)
		self.validation_dict = {'s': self.validSigAE[:nvalid], 'b': self.validBkgAE[:nvalid]}

		self.test = np.vstack((self.testSigAE[:ntest],self.testBkgAE[:ntest]))
		self.test_labels = np.array(['s'] * ntest + ['b'] * ntest)
		self.test_nlabels = np.array([1] * ntest + [0] * ntest)
		self.test_dict = {'s': self.testSigAE[:ntest], 'b': self.testBkgAE[:ntest]}

		print(f'xcheck: train/validation/test shapes: {self.train.shape}/{self.validation.shape}/{self.test.shape}')
	
	def get_kfold_validation(self,k=5,splits_total=500):
		'''
		splits_total: the max number we can divide the initial validation dataset (from splitDatasets). For the 7.2e5 dataset (default), 
		it's 500 if we want to have 288 validation samples per fold (144 Sig + 144 Bkg)

		'''
		validation_folds_sig = np.split(self.validSigAE,500)#split to folds of 288 samples: total here 500
		validation_folds_bkg = np.split(self.validBkgAE,500)
		
		validation_folds_sig = np.array(validation_folds_sig)
		validation_folds_bkg = np.array(validation_folds_bkg)
		#Create the k-fold for validation of sig+bkg equal chunks(folds) of samples:
		validation_folds = np.concatenate((validation_folds_sig,validation_folds_bkg),axis=1)
		#Return k batches of validation samples:
		return validation_folds[:k]
