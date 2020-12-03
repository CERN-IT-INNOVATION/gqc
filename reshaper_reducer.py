# Reshape events to have a single row feature vector per event to feed the AE
import numpy as np
import os
import argparse
#From format.py to map array indeces to the variable names
'''
#Variables: 
        jet_feats = ["pt","eta","phi","en","px","py","pz","btag"],
        njets = 10,
        met_feats = ["phi","pt","px","py"],
        nleps = 1,
        lep_feats = ["pt","eta","phi","en","px","py","pz"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_wq", "nMatch_tb", "nMatch_hb"],#Don't fit in AE!!
'''

def reduce_events(infile,nend,outdir=None,nstart = 0):
	#Reduce event size and apply event selection cuts
	#Using evdesc info apply phase space realistic cuts:
	#BUT we want to keep same number of signal and background events for training after the cuts.
	jets = np.load(infile+"jets.npy")
	jets = jets[:,:-3,:] #1. discard last 3 jets they are mostly initial+final state radiation
	met = np.load(infile+'met.npy')
	leps = np.load(infile+'leps.npy')
	evdesc = np.load(infile+'evdesc.npy')
	
	#Save indices for the cuts and apply numpy [indices] to all arrays
	ievents = []
	n = 0
	ind = np.logical_and(evdesc[:,2]>=2,evdesc[:,1]>=4)
	
	#event selection:
	redJet = jets[ind]
	redMet = met[ind]
	redLeps = leps[ind]
	#Old less efficient way:
	'''
	for i,iev in enumerate(evdesc):
		if iev[2]>=2 and iev[1]>=4: #njet>=4 and nbtag>=2 requirement
			ievents.append(i)
			n+=1
	print('Same:',np.array_equal(jets,redJet),np.array_equal(met,redMet),np.array_equal(leps,redLeps))
	'''
	print('Input file:',infile)
	print('Applying EVENT SELECTION: n_events = {} ---> {}'.format(jets.shape[0],redJet.shape[0]))
	
	
	print('Requested number of samples (events): {}'.format(nend-nstart))
	
	return (redJet[nstart:nend], redMet[nstart:nend], redLeps[nstart:nend])
	#Debug printing:
	#print('Reduced from: jet/met/leps shapes=',jets.shape,met.shape,leps.shape)
	#print('--->',jets_red.shape,met_red.shape,leps_red.shape)
	
	#return (jets_red, met_red, leps_red, evdesc)

def reshape_events(a):#single array or tuple(needed below for hstacking) of arrays as input
#using -1 here instead of shape[1]*shape[2] e.g. 3d Jet array -> 2d Jet array
#becayse it also includes automatically the reshaping of the non 3d arrays i.e. 2d MET
	if isinstance(a,tuple):
		i_r = []
		for i in a:
			i_r.append(i.reshape(i.shape[0],-1))
		return i_r
	else:
		a_r = a.reshape(a.shape[0],-1)
		return a_r



if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
	parser.add_argument("--nevents", type=int, default=int(2.4e5), help="number of samples to be generate. train+valid.+test")
	args = parser.parse_args()
	print('------Generating Training & Testing dataset------')
	samples = ['sig','bkg']
	n = args.nevents
	for isample in samples:
		print("_____"+isample+" samples_____")
		
		(jets,met,leps) = reduce_events(infile=isample+'_npy_sample/',nend=n)	
		feat_in = (jets,met,leps)
		
		#Map Jet btag = {0,1,...,7} --> {0,1}
		print('Map Jet btag = {0,1,...,7} (btager efficiency stages) --> {0,1} (btaged or not btaged)')#Transform to boolean
		print('btag>1 --> 1 ; else --> 0')
		jets[:,:,-1] = np.where(jets[:,:,-1]>1,1,0)
		feat_resh = reshape_events(feat_in)
		features = ["JETS","MET","LEPS"]

		for i,ifeat in enumerate(feat_resh):
			print(features[i]+":")
			print("Reshaping: {} ---> {} ".format(feat_in[i].shape,ifeat.shape))
		
		feat_out = np.hstack(feat_resh)
		print("Finalized data vectors to AE: (events,features) = {}".format(feat_out.shape))
		outdir = 'input_ae/'
		if not(os.path.exists(outdir)):
			os.mkdir(outdir)
		
		np.save(outdir+"/raw_"+isample+".npy",feat_out)
