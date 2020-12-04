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
#Simultaneous trsf for sig and bkg:
#Using numpy broadcasting without for loops for efficiency (loops run in C and not in Python)
def normalizer(bkgfile,sigfile,outfile):
	bkg,sig = np.load(bkgfile), np.load(sigfile)
	maxBkg,maxSig = np.amax(bkg,axis=0),np.amax(sig,axis=0)
	minBkg,minSig = np.amin(sig,axis=0),np.amin(bkg,axis=0)
	
	#Find global max/min for each feature for sig AND bkg
	#-->Correct normalization to retain the shapes of the pdfs
	maxTot = np.amax(np.vstack((maxBkg,maxSig)),axis=0)
	minTot = np.amin(np.vstack((minBkg,minSig)),axis=0)
	
	bkgNorm = (bkg-minTot)/(maxTot-minTot)
	sigNorm = (sig-minTot)/(maxTot-minTot)
	np.save(outfile+'Bkg.npy',bkgNorm)
	np.save(outfile+'Sig.npy',sigNorm)

infileBkg,infileSig = 'input_ae/raw_bkg.npy','input_ae/raw_sig.npy'
print('Normalizing: '+infileBkg+' & '+infileSig)
outfile = "input_ae/trainingTestingData"
normalizer(infileBkg,infileSig,outfile)
print('Output: ',outfile)


