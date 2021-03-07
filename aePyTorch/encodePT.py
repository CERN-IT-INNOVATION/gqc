import numpy as np
import torch
from aePyTorch.model import AE,tensorData 
#NB: python path naming. If encodePT is called from ../qdata.py the path is defined from the ../ hence
# aePyTorch.model is needed instead of model (even though encodePT.py is IN the aePyTorch directory)

def encode_array(data,savedModel,layers):
	dataLoader = torch.utils.data.DataLoader(tensorData(data),batch_size = data.shape[0],shuffle = False)
	device ='cpu'#for just evaluation no gpu needed...I think...
	model = AE(node_number = layers).to(device)
	model.load_state_dict(torch.load(savedModel+'bestModel.pt',map_location=torch.device('cpu')))
	model.eval()

	with torch.no_grad():
		dataIter = iter(dataLoader)
		inp = dataIter.next().to(device)
		_, output = model(inp.float())
		return output.cpu().numpy()

def encode(data,savedModel,layers):
	if (isinstance(data, dict)):
		for x in data:
			x = torch.Tensor(x)
			data[x] = encode_array(data[x],savedModel,layers)
		return data
	
	return encode_array(torch.Tensor(data),savedModel,layers)
