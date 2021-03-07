import numpy as np
from aePyTorch.splitDatasets import splitDatasets

#TODO: Takes some time to load encoded datasets, because the AE model runs on the full datasets and then outputs only a subset 

class qdata:
	'''
	Class from with which the training, validation and testing datasets of quantum classifiers is defined
	

	'''

	#Load the test dataset of the autoencoder and split it to training,validation and testing datasets for the qml
	#infiles = ('input_ae/trainingTestingDataSig.npy','input_ae/trainingTestingDataBkg.npy')
	infiles = ('input_ae/trainingTestingDataSig7.2e5.npy','input_ae/trainingTestingDataBkg7.2e5.npy')
	trainSigAE, trainBkgAE, validSigAE, validBkgAE, testSigAE,testBkgAE = splitDatasets(infiles,separate=True, not_all = False)
	
	ntot_test = int(testSigAE.shape[0])
	ntot_train = int(trainSigAE.shape[0])
	ntot_valid = int(validSigAE.shape[0])

	def __init__(self, encoder = "", train_p = 0.0005, valid_p = 0.005, 
				test_p = 0.005):
		'''	   
		Args:	
    	----------
    	train_p : float or int
			float: Proportion of total training data set (from splitDatasets) that will be used in the training of the quantum
			classifiers and the classical models used for benchmarking (trained and tested on the same data sets)
			int: Number of training data samples to be used
    	valid_p : float
			float: Proportion of total validation data set (from splitDatasets) that will be used in the training of the quantum
			classifiers and the classical models used for benchmarking (trained and tested on the same data sets)
			int: Number of validation data samples to be used
    	test_p : float
			float: Proportion of total testing data set (from splitDatasets) that will be used in the training of the quantum
			classifiers and the classical models used for benchmarking (trained and tested on the same data sets)
			int: Number of testing data samples to be used
    	-------
		'''
		#TODO: maybe we should go **kwargs, so we don't have a lot of argumnets
		if encoder == "tf":
			from aeTF.encode import encode
			from aeTF.encode import model
			print('Using TensorFlow for autoencoder model')
			self.trainSigAE = encode(self.trainSigAE)
			self.trainBkgAE = encode(self.trainBkgAE)
			self.validSigAE = encode(self.validSigAE)
			self.validBkgAE = encode(self.validBkgAE)
			self.testSigAE = encode(self.testSigAE)
			self.testBkgAE = encode(self.testBkgAE)
			self.model = model
		elif encoder == "pt":
			from encodePT import encode
			print('Using PyTorch for autoencoder model to encode the data')
			#TODO: add functionality: use other models for autoencoder
			self.savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr2e-037.2e5/"
			self.layers = [67,64, 52, 44, 32, 24, 16]
			self.trainSigAE = encode(self.trainSigAE,self.savedModel,self.layers)
			self.trainBkgAE = encode(self.trainBkgAE,self.savedModel,self.layers)
			self.validSigAE = encode(self.validSigAE,self.savedModel,self.layers)
			self.validBkgAE = encode(self.validBkgAE,self.savedModel,self.layers)
			self.testSigAE = encode(self.testSigAE,self.savedModel,self.layers)
			self.testBkgAE = encode(self.testBkgAE,self.savedModel,self.layers)

		elif encoder == "":
			print("Using unencoded data")
		else:
			raise Exception('Unknown encoder')


		if train_p <= 1:
			ntrain = int(self.ntot_train*train_p)
			self.train_p = train_p
			if self.ntot_train % ntrain != 0:
				raise Exception('ntot_train mod ntrain != 0, choose train_p so the dataset can be divided')
		else:
			ntrain = train_p
			self.train_p = train_p / self.ntot_train

		if valid_p <= 1:
			nvalid = int(self.ntot_valid*valid_p)
			self.valid_p = valid_p
			if self.ntot_valid % nvalid != 0:
				raise Exception('ntot_valid mod nvalid != 0, choose valid_p so the dataset can be divided')
		else:
			nvalid = valid_p
			self.valid_p = valid_p / self.ntot_valid
		
		if (test_p <= 1) & (test_p > 0):
			ntest = int(self.ntot_test*test_p)
			self.test_p = test_p
			if self.ntot_test % ntest != 0:
				raise Exception('ntot_test mod ntest != 0, choose test_p so the dataset can be divided')
		else:
			ntest = test_p
			self.test_p = test_p / self.ntot_test

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
	
	def get_kfold_validation(self,k=5):
		splits_total = int(1/self.valid_p)
		'''
		splits_total: the max number we can divide the initial validation dataset (from splitDatasets). 
		E.g. for the 7.2e5 dataset (default),  it's 500 if we want to have 288 validation samples per fold (144 Sig + 144 Bkg)
		'''
		validation_folds_sig = np.split(self.validSigAE,splits_total)
		validation_folds_bkg = np.split(self.validBkgAE,splits_total)
		
		validation_folds_sig = np.array(validation_folds_sig)
		validation_folds_bkg = np.array(validation_folds_bkg)
		#Create the k-fold for validation of sig+bkg equal chunks(folds) of samples:
		validation_folds = np.concatenate((validation_folds_sig,validation_folds_bkg),axis=1)
		#Return k batches of validation samples:
		return validation_folds[:k]#if k>splits_total it will just return the [:splits_total] folds
