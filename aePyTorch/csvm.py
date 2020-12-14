import numpy as np
import warnings
import torch
from model import * #Load custom model
from splitDatasets import *
import argparse
from sklearn import svm


#########################TODO use encode module to load the latent data:
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infiles = ('../input_ae/trainingTestingDataSig.npy','../input_ae/trainingTestingDataBkg.npy')

savedModel = "trained_models/L64.52.44.32.24.16B128Lr3e-03MoreDataNewNorm/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--input", type=str, default=infiles, nargs =2, help="path to datasets")
parser.add_argument('--model',type=str,required=True,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')

args = parser.parse_args()
infiles = args.input
(savedModel,layers) = (args.model,args.layers)

_,validDataset,testDataset,_,validLabels, testLabels = splitDatasets(infiles, labels=True)#Load test samples
testLoader = torch.utils.data.DataLoader(arrayData(testDataset),batch_size = testDataset.shape[0],shuffle = False)
validLoader = torch.utils.data.DataLoader(arrayData(validDataset),batch_size = validDataset.shape[0],shuffle = False)

feature_size = testDataset.shape[1]
layers.insert(0,feature_size)

model = AE(node_number = layers).to(device)
model.load_state_dict(torch.load(savedModel+'bestModel.pt'))
model.eval()

testDataSig, testDataBkg = splitDatasets(infiles,separate=True)

#Model output for the SVM training input:
with torch.no_grad():
	dataIter = iter(testLoader)
	inp = dataIter.next()
	_, testOutput = model(inp.float())
	dataIter = iter(validLoader)
	inp = dataIter.next()
	_, validOutput = model(inp.float())
################################################################################################


#TODO save svm results in SVM log automatically
#TODO create accessible Ntrain variable, and benchmark qsvm with csvm give the same training size
train = np.array(validOutput)
labels = np.array(validLabels)

test = np.array(testOutput)
tlabels = np.array(testLabels)

cls = svm.SVC()
cls.fit(train, labels);

res = np.array(cls.predict(test))

print("Success ratio: ", sum(res == tlabels) / len(res))

