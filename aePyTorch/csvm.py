import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import torch
import torch.nn as nn
from model import * #Load custom model
from splitDatasets import *
import argparse


#use gpu if available
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infiles = ('../input_ae/trainingTestingDataSig.npy','../input_ae/trainingTestingDataBkg.npy')

savedModel = "trained_models/L64.52.44.32.24.16B128Lr3e-03MoreDataNewNorm/"
layers = [64, 52, 44, 32, 24, 16]

_,validDataset,testDataset,validLabels, testLabels = splitDatasets(infiles, labels=True)#Load test samples
testLoader = torch.utils.data.DataLoader(arrayData(testDataset),batch_size = testDataset.shape[0],shuffle = False)
validLoader = torch.utils.data.DataLoader(arrayData(validDataset),batch_size = validDataset.shape[0],shuffle = False)

feature_size = testDataset.shape[1]
layers.insert(0,feature_size)

model = AE(node_number = layers).to(device)
model.load_state_dict(torch.load(savedModel+'bestModel.pt'))
model.eval()

criterion = nn.MSELoss(reduction= 'mean')

#if __name__ == "__main__":
testDataSig, testDataBkg = splitDatasets(infiles,separate=True)
testLoaderSig = torch.utils.data.DataLoader(arrayData(testDataSig),batch_size = testDataSig.shape[0],shuffle = False)
testLoaderBkg = torch.utils.data.DataLoader(arrayData(testDataBkg),batch_size = testDataBkg.shape[0],shuffle = False)

#Latent pdf's & input-vs-output pdf's:
with torch.no_grad():
	#MSE on whole validation & test samples:
	dataIter = iter(testLoader)
	inp = dataIter.next()
	_, testOutput = model(inp.float())
	dataIter = iter(validLoader)
	inp = dataIter.next()
	_, validOutput = model(inp.float())


from sklearn import svm


train = np.array(validOutput)
labels = np.array(validLabels)

test = np.array(testOutput)
tlabels = np.array(testLabels)

cls = svm.SVC()
cls.fit(train, labels);

res = np.array(cls.predict(test))

print("Success ratio: ", sum(res == tlabels) / len(res))

