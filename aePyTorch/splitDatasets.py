import numpy as np

#Gets the full dataset and splits it into training, validation and test datasets
#returns numpy arrays
def splitDatasets(infiles: tuple,separate = False):
	#separate: flag to return Bkg and Sig seperately. Used for the test samples and pdf plots.
	#(Sig,Bkg) file
	dataArraySig = np.load(infiles[0])
	dataArrayBkg = np.load(infiles[1])
	
	#ntot each bkg & sig
	if dataArraySig.shape[0] != dataArrayBkg.shape[0]:
	        raise Exception('nSig != nBkg! Events should be equal')
	ntot = int(dataArraySig.shape[0])
	ntrain, nvalid, ntest = int(ntot*0.8), int(0.1*ntot), int(0.1*ntot)
	print('splitDatasets.py:')
	print('xcheck: (ntrain={}, nvalid={}, ntest={})x2; for Sig & Bkg'.format(ntrain,nvalid,ntest))
	
	#Training samples:
	dataset = np.vstack((dataArraySig[:ntrain],dataArrayBkg[:ntrain]))
	#Validation samples:
	validDataset = np.vstack((dataArraySig[ntrain:(ntrain+nvalid)],dataArrayBkg[ntrain:(ntrain+nvalid)]))
	#Testing samples:
	testDataset = np.vstack((dataArraySig[(ntrain+nvalid):],dataArrayBkg[(ntrain+nvalid):]))
	
	print(f'features = {dataset.shape[1]}, ntrain={dataset.shape[0]}, nvalid={validDataset.shape[0]}, ntest={testDataset.shape[0]});Sig + Bkg samples\n')
	if np.array_equal(validDataset,testDataset):
		raise Exception('Validation and Testing datasets are the same!')
	
	if separate:
		testSigDataset,testBkgDataset = np.vsplit(testDataset,2)
		return testSigDataset,testBkgDataset

	return dataset, validDataset, testDataset

