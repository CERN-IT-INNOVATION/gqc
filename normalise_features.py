import numpy as np
#Normalise all features to [0,1] to see how it affects AE learning 
'''
#Variables: 
        jet_feats = ["pt","eta","phi","en","px","py","pz","btag"],
        njets = 10,
        met_feats = ["phi","pt","px","py"],
        nleps = 1,
        lep_feats = ["pt","eta","phi","en","px","py","pz"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_wq", "nMatch_tb", "nMatch_hb"],#Don't fit in AE!!
'''
def normalizer(infile,outfile):
	dataset = np.load(infile)
	dataset_normed = np.zeros_like(dataset.T)
	
	for i,ifeature in enumerate(dataset.T):#to iterate over the columns i.e. features and not events
		xmax, xmin = np.amax(ifeature), np.amin(ifeature)
		if xmax != xmin:#not cause NaN's
			ifeature_normed = (ifeature-xmin)/(xmax-xmin)
			dataset_normed[i] = ifeature_normed
		else:
			dataset_normed[i] = ifeature
	#for i,ifeature in enumerate(dataset_normed):
	#	print(ifeature)
	np.save(outfile,dataset_normed.T)
	return None

#Training
print('Training + Testing Samples')
infile = "input_ae/raw_sig.npy"
outfile = "input_ae/trainingTestingDataSig.npy"
print('Normalizing: '+infile)
normalizer(infile,outfile)
print('Output: ',outfile)

infile = "input_ae/raw_bkg.npy"
outfile = "input_ae/trainingTestingDataBkg.npy"
print('Normalizing: '+infile)
normalizer(infile,outfile)
print('Output: ',outfile)

#
