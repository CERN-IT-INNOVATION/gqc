import numpy as np
import warnings
import torch
from aePyTorch.model import * #Load custom model

#with warnings.catch_warnings():
#	warnings.simplefilter("ignore")
#	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#FIXME: If run on a GPU an error in Pytorch is raised that some part of the compution is loaded on cpu while some part is on gpu. Find source of error.
device='cpu'
def encode_array(data,savedModel,layers):
	dataLoader = torch.utils.data.DataLoader(arrayData(data),batch_size = data.shape[0],shuffle = False)

	model = AE(node_number = layers).to(device)
	model.load_state_dict(torch.load(savedModel+'bestModel.pt'))
	model.eval()

	with torch.no_grad():
		dataIter = iter(dataLoader)
		inp = dataIter.next()
		_, output = model(inp.float())
		return np.array(output);

def encode(data,savedModel,layers):
	if (isinstance(data, dict)):
		for x in data:
			data[x] = encode_array(data[x],savedModel,layers)
		return data;
	
	return encode_array(data,savedModel,layers);	
