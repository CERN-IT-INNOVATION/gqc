import numpy as np
import torch
import torch.nn as nn

#Load numpy arrays as data
class tensorData(torch.utils.data.Dataset):
	def __init__(self,x):
		self.x = torch.Tensor(x) #x is the numpy array dataset #TODO float casting
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

'''
TIPS for PyTorch (1.7.0):

-When a tensor is created it is automatically loaded on cpu. If gpu is desired to(device) needs to be cast on every tensor. 
- cpu->gpu loading requires copying the tensor elements which can take time the benefit of using gpu comes after loading during
  matrix/tensor parallelized manipulations
- A custom model class inheriting from nn.Module is "a list" of tensors (called also parameters of the model) tha are trainable, i.e. gradients
  are computed by pytorch. If one defines a custom layer/tensor which should be a trainable part of the model on should call a nn.Parameter
  wrapper to make it part of the parameter list of the model.
- When calling model.to(device) all the parameters of the model ara loaded to the device (if one does not call nn.Parameter on the custom tensors
  they will not be loaded to the device, potentially causing problems)
Example:
self.Min = nn.Parameter(torch.Tensor([Min]), requires_grad=False) 
model = Model() 
model.Min 
print(model) -> Whatever is printed is part of the model parameters (nn.Parameter object)

'''