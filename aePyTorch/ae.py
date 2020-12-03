import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import * 
from train import *
from splitDatasets import *
import warnings #Supress GPU not existing warnings
import argparse

seed = 100
torch.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)#autograd error check

#use gpu if available
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:',device)
infiles = ('../input_ae/trainingTestingDataSig.npy','../input_ae/trainingTestingDataBkg.npy')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--input", type=str, default=infiles, nargs =2, help="path to datasets")
parser.add_argument('--lr',type=float,default=2e-03,help='learning rate')
parser.add_argument('--layers',type=int,default=[64,44,32,24,16],nargs='+',help='type hidden layers node number')
parser.add_argument('--batch',type=int,default=64,help='batch size')
parser.add_argument('--epochs',type=int,default=85,help='number of training epochs')
parser.add_argument('--fileFlag',type=str,default='',help='fileFlag to concatenate to filetag')

args = parser.parse_args()
infiles = args.input
(learning_rate,layers,batch_size,epochs,fileFlag) = (args.lr,args.layers,args.batch,args.epochs,args.fileFlag)
#print(learning_rate,layers,batch_size,epochs,fileFlag) 

#Sig+Bkg training with shuffle
dataset, validDataset,_ = splitDatasets(infiles)

feature_size = dataset.shape[1]
layers.insert(0,feature_size)#insert at the beginning of the list the input dim.
validation_size = validDataset.shape[0]

#Convert to torch dataset:
dataset = arrayData(dataset) 
validDataset = arrayData(validDataset)

train_loader = torch.utils.data.DataLoader(dataset,batch_size = args.batch,shuffle = True)
valid_loader = torch.utils.data.DataLoader(validDataset,batch_size = validation_size,shuffle = True)

model = AE(node_number = layers).to(device)

print('Training...')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#create an optimizer object
#betas=(0.9, 0.999) #play with the decay of the learning rate for better results
criterion = nn.MSELoss(reduction= 'mean')#mean-squared error loss

print('Batch size ='+str(batch_size)+', learning_rate='+str(learning_rate)+', layers='+str(model.node_number))

#Prepare to save training outputs:
layersTag = '.'.join(str(inode) for inode in model.node_number[1:])#Don't print input size
filetag = 'L'+layersTag+'B'+str(batch_size)+'Lr{:.0e}'.format(learning_rate)+fileFlag#only have 1 decimal lr

outdir = '/data/vabelis/disk/sample_preprocessing/aePyTorch/trained_models/'+filetag+'/'
if not(os.path.exists(outdir)):
	os.mkdir(outdir)

#Call training function:
lossTrainValues,lossValidValues,minValid = train(train_loader,valid_loader,model,criterion,optimizer,epochs,device,outdir)

#Loss -vs- epochs plot
plt.plot(list(range(epochs)),lossTrainValues,label='Training Loss (last batch)')
plt.plot(list(range(epochs)),lossValidValues,label='Validation Loss (1 per epoch)')
plt.ylabel("MSE")
plt.xlabel("epochs")
plt.title("B ="+str(batch_size)+", lr="+str(learning_rate)+", "+str(model.node_number)+', L ={:.6f}'.format(minValid))
plt.legend()

#Save loss plot:
plt.savefig(outdir+'lossVSepochs.png')

#Save MSE log to check best model:
with open('trained_models/mseLog.txt','a+') as mseLog:
	logEntry = filetag+': Min. Validation loss = {:.6f}\n'.format(minValid)
	mseLog.write(logEntry)
