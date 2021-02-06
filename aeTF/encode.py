from tensorflow.keras.models import load_model
import numpy as np

model = "F-8-30-8-F3"

ae = load_model("aeTF/out/" + model + "/model")

def encode(data):
	if (isinstance(data, dict)):
		for x in data:
			data[x] = np.array(ae.encoder(np.copy(data[x])))
		return data;
	
	return np.array(ae.encoder(np.copy(data)));
