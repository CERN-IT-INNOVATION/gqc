import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import torch
import torch.nn as nn
from model import AE,tensorData #Load custom model
from splitDatasets import splitDatasets
import argparse
from varname import *

#give 2 1d arrays
def ratioPlotter(inp,out,ifeature,classLabel = ''):
	plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
	plt.rc('axes', titlesize=22)     # fontsize of the axes title
	plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
	plt.rc('legend', fontsize=22)    # legend fontsize
	#plt.rc('figure', titlesize=22)  # fontsize of the figure title	
	hIn,binsIn, patchIn = plt.hist(x=inp,bins=60,range =(0,1),alpha=0.8,histtype='step',linewidth=2.5,label=classLabel,density = True)
	hOut,binsOut, patch = plt.hist(x=out,bins=60,range = (0,1),alpha=0.8,histtype='step',linewidth = 2.5,label='Rec. '+classLabel, density = True)
	plt.xlabel(varname(i)+' (normalized)')
	plt.ylabel('Density')
	plt.xlim(0,0.4)
	#plt.title('Distribution of '+varname(i))
	plt.legend()


#use gpu if available
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infiles = ('../input_ae/trainingTestingDataSig.npy','../input_ae/trainingTestingDataBkg.npy')
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--input", type=str, default=infiles, nargs =2, help="path to datasets")
parser.add_argument('--model',type=str,required=True,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')
parser.add_argument('--batch',type=int,default=128,help='batch size for testing histogram plot')

args = parser.parse_args()
infiles = args.input
(savedModel,layers,batch_size) = (args.model,args.layers,args.batch)

_,validDataset,testDataset = splitDatasets(infiles)#Load test samples
testLoader = torch.utils.data.DataLoader(tensorData(testDataset),batch_size = testDataset.shape[0],shuffle = False)
validLoader = torch.utils.data.DataLoader(tensorData(validDataset),batch_size = validDataset.shape[0],shuffle = False)

feature_size = testDataset.shape[1]
layers.insert(0,feature_size)

model = AE(node_number = layers).to(device)
model.load_state_dict(torch.load(savedModel+'bestModel.pt',map_location=torch.device('cpu')))
model.eval()

criterion = nn.MSELoss(reduction= 'mean')

#if __name__ == "__main__":
testDataSig, testDataBkg = splitDatasets(infiles,separate=True)
testLoaderSig = torch.utils.data.DataLoader(tensorData(testDataSig),batch_size = testDataSig.shape[0],shuffle = False)
testLoaderBkg = torch.utils.data.DataLoader(tensorData(testDataBkg),batch_size = testDataBkg.shape[0],shuffle = False)
def mape(output,target,epsilon=1e-4):
	loss = torch.mean(torch.abs((output-target)/(target+epsilon)))
	return loss

#Latent pdf's & input-vs-output pdf's:
with torch.no_grad():
	#MSE on whole validation & test samples:
	dataIter = iter(testLoader)
	inp = dataIter.next()
	output, latentOutput = model(inp.float())
	print('Test sample MSE:',criterion(output,inp).item())
	
	dataIter = iter(validLoader)
	inp = dataIter.next()
	output, latentOutput = model(inp.float())
	print('Validation sample MSE:',criterion(output,inp).item())
	
	dataIter = iter(testLoader)
	criterion = mape
	inp = dataIter.next()
	output, latentOutput = model(inp.float())
	print('Test sample MAPE:',criterion(output,inp).item())
	
	#Latent pdf's:
	dataIter = iter(testLoaderSig)
	inpSig = dataIter.next()
	outputSig, latentOutputSig = model(inpSig.float())
	dataIter = iter(testLoaderBkg)
	inpBkg = dataIter.next()
	outputBkg, latentOutputBkg = model(inpBkg.float())
	latentOutputSig,latentOutputBkg = latentOutputSig.numpy(), latentOutputBkg.numpy()
	
	for i in range(latentOutputSig.shape[1]):
		xmax = max(np.amax(latentOutputSig[:,i]),np.amax(latentOutputBkg[:,i]))
		xmin = min(np.amin(latentOutputSig[:,i]),np.amin(latentOutputBkg[:,i]))
		hSig,_,_ = plt.hist(x=latentOutputSig[:,i],density=1,range = (xmin,xmax),bins=50,alpha=0.6,histtype='step',linewidth=2.5,label='Sig')
		hBkg,_,_ = plt.hist(x=latentOutputBkg[:,i],density=1,range = (xmin,xmax),bins=50,alpha=0.6,histtype='step',linewidth=2.5,label='Bkg')
		plt.legend()
		plt.xlabel(f'Latent feature {i}')
		plt.savefig(savedModel+'latentPlot'+str(i)+'.png')
		plt.clf()	

	#Input VS Output:
	for i in range(inpSig.numpy().shape[1]):
		plt.figure(figsize=(12,10))
		ratioPlotter(inpBkg.numpy()[:,i],outputBkg.numpy()[:,i],i,classLabel='Background')#Plot Background distributions
		ratioPlotter(inpSig.numpy()[:,i],outputSig.numpy()[:,i],i,classLabel='Signal')#Plot Signal distributions
		
		plt.savefig(savedModel+'ratioPlot'+varname(i)+'.pdf')
		plt.clf()
	
	#FIXME: Not displayed properly. Values are off and seems to be bias still between sig and bkg for the MSE distributions
	##MSE loss in test samples.
	samples = ['Sig','Bkg']
	testLoaderSig = torch.utils.data.DataLoader(tensorData(testDataSig),batch_size = batch_size,shuffle = True)
	testLoaderBkg = torch.utils.data.DataLoader(tensorData(testDataBkg),batch_size = batch_size,shuffle = True)
	loaders = [testLoaderSig,testLoaderBkg]
	colors = ['b','r']
	labels = ['Test on Sig.', 'Test on Bkg.']
	for j,isample in enumerate(samples):
		meanLossBatch = []
		for i,batch_features in enumerate(loaders[j]):
			batch_features = batch_features.view(-1, feature_size).to(device)
			output,_ = model(batch_features)
			loss = criterion(output,batch_features)
			meanLossBatch.append(loss.item())
		#Fix the range and normalization of the below hist:
		plt.hist(meanLossBatch,bins=20,density = 0,color=colors[j],alpha=0.5, ec='black',label=labels[j])
		plt.ylabel('Entries/Bin')
		plt.xlabel('MSE per Batch')
		plt.title('MSE per batch, Ntest={}.'.format(len(testDataset)))
	plt.legend()
	plt.savefig(savedModel+'testLossHist.png',dpi=300)

