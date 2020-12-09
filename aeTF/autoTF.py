import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

from compare import *
from data import *


class Autoencoder(Model):
	def __init__(self, latent_dim, object_dim, nlayers):
		super().__init__();
		self.latent_dim = latent_dim;
		
		self.encoder = tf.keras.Sequential();

		if type(nlayers) != list:
			layernums = [];
			num_layers = nlayers;
			for i in range(num_layers):
				layernums.append(int(object_dim - i * (object_dim - latent_dim)/num_layers));
			layernums.append(latent_dim)
		else:
			layernums = nlayers;
			num_layers = len(nlayers);

		for i in range(num_layers - 1):
			self.encoder.add(layers.Dense(layernums[i], activation = 'elu'))
			print(layernums[i])
		self.encoder.add(layers.Dense(latent_dim, activation = 'elu'))

		self.decoder = tf.keras.Sequential()

		for i in range(num_layers -1):
			self.decoder.add(layers.Dense(layernums[num_layers - i - 1], activation = 'elu'))
			print(layernums[num_layers - i - 1])
		self.decoder.add(layers.Dense(object_dim, activation = 'sigmoid'))


	def call(self, x):
		encoded = self.encoder(x);
		decoded = self.decoder(encoded);
		return decoded;



def compute(nlayers, verb = 0, name = "E", lsdim = 16, ep = 30, factor = None):
	
	bs = 32
	lr = 1e-3

	if (factor != None):
		bs = int(bs * factor);
		lr = lr * np.sqrt(factor);
	
	autoencoder = Autoencoder(lsdim, train.shape[1], nlayers)
	myAdam = Adam(learning_rate = lr)
	autoencoder.compile(optimizer = myAdam, loss=losses.MeanSquaredError());

	history = autoencoder.fit(train, train, epochs = ep, shuffle = True, validation_data = (vali, vali), verbose = verb, batch_size = bs);
	error = np.mean(MSE(test, autoencoder.call(test)))

	if (name != None):
		name = name + "-" + str(nlayers) + "-" + str(ep) + "-" +str(lsdim);
		if (factor != None):
			name = name + "-F" + str(factor);
		if not (os.path.isdir("out")):
			os.mkdir("out")
		os.mkdir("out/" + name)
		save_model(autoencoder, "out/"+ name + "/model")
		final_bkg = autoencoder.decoder(autoencoder.encoder(test_bkg));
		final_sig = autoencoder.decoder(autoencoder.encoder(test_sig));
		encoded_bkg = autoencoder.encoder(test_bkg)
		encoded_sig = autoencoder.encoder(test_sig)

		errs = history.history['loss'];
		plt.figure();
		plt.plot(range(len(errs)), errs)
		plt.title("MSE for " + name)
		plt.xlabel("Epochs")
		plt.savefig("out/" + name + "/error.png");

		compare(test_bkg, final_bkg, encoded_bkg, test_sig, final_sig, encoded_sig, name, True)
	
	return error, autoencoder, history
