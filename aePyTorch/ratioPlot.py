import numpy as np
from compare import *
import matplotlib.pyplot as plt
import os
import torch
from model import * #Load custom model

#give 2 1d arrays
def ratioPlotter(inp,out,classLabel = '',ifeature):
	hIn,binsIn, patchIn = plt.hist(x=inp,bins=60,range =(0,1),alpha=0.5,histtype='step',linewidth=2.5,label='Original '+classLabel)
	hOut,binsOut, patch = plt.hist(x=out,bins=60,range = (0,1),alpha=0.5,histtype='step',linewidth = 2.5,label='Output '+classLabel)
	plt.xlabel('feature '+ifeature)
	plt.ylabel('Entries/Bin')
	plt.title('Distribuition of ')
	plt.legend()
	#NB: maybe use Samuel's ifeature->varname function for presentation


if __name__ == "__main__":
	ntrain = int(1e5)
	testDataBkg = np.load('../input_ae/trainingTestingDataBkg.npy')
	testDataBkg = testDataBkg[ntrain:]
	
	testDataSig = np.load('../input_ae/trainingTestingDataSig.npy')
	testDataSig = testDataSig[ntrain:]
	
	feature_size = testDataBkg.shape[1]
	#use gpu if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	#model = AE(node_number = [feature_size,97,96,95],input_shape=feature_size).to(device)
	model = AE(input_shape = feature_size).to(device)
	model.load_state_dict(torch.load("trained_models/trained_modelSigBkg8Latent32BatchELUSNewFeat.pt"))
	model.eval()
	
	#Check for Sig only:
	#put batch_size = whole test dataset to just calculate the network outputs at once
	testLoaderSig = torch.utils.data.DataLoader(arrayData(testDataSig),batch_size = testDataSig.shape[0],shuffle = False)
	testLoaderBkg = torch.utils.data.DataLoader(arrayData(testDataBkg),batch_size = testDataBkg.shape[0],shuffle = False)
	
	with torch.no_grad():#Disabling gradient calculation. No use of Tensor.backward() here, so no need to save gradients.
		dataIter = iter(testLoaderSig)
		inpSig = dataIter.next()
		outputSig = model(inpSig.float())
		dataIter = iter(testLoaderBkg)
		inpBkg = dataIter.next()
		outputBkg = model(inpBkg.float())
	
		for i in range(inpSig.numpy().shape[1]):
			ratioPlotter(inpSig.numpy()[:,i],outputSig.numpy()[:,i],classLabel='Signal')#Plot Signal distributions
			ratioPlotter(inpBkg.numpy()[:,i],outputBkg.numpy()[:,i],classLabel='Background')#Plot Background distributions
			outdir = '/afs/cern.ch/user/v/vabelis/ratioPlots'+str(model.node_number[-1])+'latentNewFeatNormed'
			if not(os.path.exists(outdir)):
				os.mkdir(outdir)
			plt.savefig(outdir+'/ratioPlot'+str(i)+'.png')
			plt.clf()
