import numpy as np
import torch
import torch.nn as nn

#Load numpy arrays as data
class arrayData(torch.utils.data.Dataset):
	def __init__(self,x):
		self.x = torch.Tensor(x) #x is the numpy array dataset
		#self.x_labels = x_labels No labels here, we want autoencoder
	def __len__(self):
		return len(self.x) #same as self.x.shape[0]
	def __getitem__(self,index):
		return self.x[index] #self.x_labels[index] if existing
###Model definition:
class AE(nn.Module):
	def __init__(self,node_number,dropout=False,**kwargs):#input layer included in node_number
		super(AE, self).__init__()
		self.node_number = node_number
		
		self.encoderLayers = []
		#self.encoderLayers.append(nn.BatchNorm1d(node_number[0]))
		#self.encoderLayers.append(nn.InstanceNorm1d(node_number[0]))
		for i,inodes in enumerate(node_number):
			if dropout == True:
				if i != 0:
					prob=0.2
					self.encoderLayers.append(nn.Dropout(p=prob))
			self.encoderLayers.append(nn.Linear(node_number[i],node_number[i+1]))
			if i == len(node_number)-2: #break when i+1 is final index
				#self.encoderLayers.append(nn.Softmax())
				self.encoderLayers.append(nn.Sigmoid())
				break	
			self.encoderLayers.append(nn.ELU(True))
		self.encoder = nn.Sequential(*self.encoderLayers)
		
		self.decoderLayers = []
		for j,jnodes in reversed(list(enumerate(node_number))):
			if dropout == True:	
				prob=0.2
				self.decoderLayers.append(nn.Dropout(p=prob))
			
			self.decoderLayers.append(nn.Linear(node_number[j],node_number[j-1]))
			if j == 1:#break when we reach output layer with j-1 == 0
				break
			self.decoderLayers.append(nn.ELU(True))
		self.decoderLayers.append(nn.Sigmoid())
		#self.decoderLayers.append(nn.ELU(True))
		self.decoder = nn.Sequential(*self.decoderLayers)
		
	def forward(self, x):
		latent = self.encoder(x)
		reconstructed = self.decoder(latent)
		return reconstructed, latent

