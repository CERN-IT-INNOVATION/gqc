import numpy as np
from model import * #Load custom model


#use gpu if available
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

savedModel = "trained_models/L64.52.44.32.24.16B128Lr3e-03MoreDataNewNorm/"
layers = [64, 52, 44, 32, 24, 16]

def encode(my_data):
	dataLoader = torch.utils.data.DataLoader(arrayData(my_data),batch_size = my_data.shape[0],shuffle = False)
	feature_size = my_data.shape[1]
	layers.insert(0,feature_size)

	model = AE(node_number = layers).to(device)
	model.load_state_dict(torch.load(savedModel+'bestModel.pt'))
	model.eval()

	with torch.no_grad():
		dataIter = iter(dataLoader)
		inp = dataIter.next()
		_, output = model(inp.float())
		return output;
