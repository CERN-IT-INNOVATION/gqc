import numpy as np
import torch
from aePyTorch.model import AE,tensorData 

def encode_array(data,savedModel,layers):
	dataLoader = torch.utils.data.DataLoader(tensorData(data),batch_size = data.shape[0],shuffle = False)
	device ='cpu'#for just evaluation no gpu needed...I think...
	model = AE(node_number = layers).to(device)
	model.load_state_dict(torch.load(savedModel+'bestModel.pt'))
	model.eval()

	with torch.no_grad():
		dataIter = iter(dataLoader)
		inp = dataIter.next().to(device)
		_, output = model(inp.float())
		return output.cpu().numpy()

def encode(data,savedModel,layers):
	if (isinstance(data, dict)):
		for x in data:
			data[x] = encode_array(data[x],savedModel,layers)
		return data;
	
	return encode_array(data,savedModel,layers);	
