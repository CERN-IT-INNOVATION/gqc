import numpy as np
import torch
from splitDatasets import splitDatasets
import argparse, time
from sklearn import svm
from encodePT import encode

start_time = time.time()

infiles = ('../input_ae/trainingTestingDataSig.npy','../input_ae/trainingTestingDataBkg.npy')
savedModel = "trained_models/L64.52.44.32.24.16B128Lr3e-03SigmoidLatent/"
defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--input", type=str, default=infiles, nargs =2, help="path to datasets")
parser.add_argument('--model',type=str,default=savedModel,help='path to saved model')
parser.add_argument('--layers',type=int,default=defaultlayers,nargs='+',help='type hidden layers nodes corresponding to saved model')

args = parser.parse_args()
infiles = args.input
(savedModel,layers) = (args.model,args.layers)


_,validDataset,testDataset,_,validLabels, testLabels = splitDatasets(infiles, labels=True)
feature_size = testDataset.shape[1]
layers.insert(0,feature_size)

#TODO save svm results in SVM log automatically
#TODO Use also qdata to have fair benchmark with qclassifiers
#TODO Have accessible Ntrain variable

train = encode(validDataset,savedModel,layers)
labels = np.array(validLabels)

test = encode(testDataset,savedModel,layers)
tlabels = np.array(testLabels)

print(f'Training samples for SVM: {len(train)}.')
print(f'Testing samples for SVM: {len(test)}.')

cls = svm.SVC()
cls.fit(train, labels);

res = np.array(cls.predict(test))

print("Success ratio: ", sum(res == tlabels) / len(res))

end_time = time.time()
print(f"Execution Time {end_time-start_time} s or {(end_time-start_time)/60} min.")

