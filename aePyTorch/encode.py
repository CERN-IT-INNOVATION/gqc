import numpy as np
import warnings
import torch
import torch.nn as nn
from aePyTorch.model import * #Load custom model


#use gpu if available
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

savedModel = "aePyTorch/trained_models/L64.52.44.32.24.16B128Lr3e-03MoreDataNewNorm/"
layers = [64, 52, 44, 32, 24, 16]
feature_size = 67
layers.insert(0,feature_size)

def encode_array(my_data):
	dataLoader = torch.utils.data.DataLoader(arrayData(my_data),batch_size = my_data.shape[0],shuffle = False)

	model = AE(node_number = layers).to(device)
	model.load_state_dict(torch.load(savedModel+'bestModel.pt'))
	model.eval()

	with torch.no_grad():
		dataIter = iter(dataLoader)
		inp = dataIter.next()
		_, output = model(inp.float())
		return np.array(output);

def encode(my_data):
	if (isinstance(my_data, dict)):
		for x in my_data:
			my_data[x] = encode_array(my_data[x])
		return my_data;
	
	return encode_array(my_data);	
