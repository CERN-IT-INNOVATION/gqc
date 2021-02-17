import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import save_model

import os
import matplotlib.pyplot as plt

from vqctf.circuit import *

import time
from datetime import datetime

def get_labels(model, data):
	data = np.array(model.predict(data))
	for i in range(len(data)):
		if (data[i] < 0.5):
			data[i] = 0
		else:
			data[i] = 1
	return data

def get_accuracy(model, data, real):
	data = get_labels(model, data)
	return sum(data == real) / len(data)


class CustomCallback(keras.callbacks.Callback):
	def __init__(self, qd):
		self.qd = qd

#	def on_epoch_end(self, epoch, logs=None):	
		#train_acc = get_accuracy(self.model, self.qd.train, self.qd.train_nlabels)
		#valid_acc = get_accuracy(self.model, self.qd.validation, self.qd.validation_nlabels)
		#print(f"Epoch {epoch} | Train accuracy: {train_acc} | Valid accuracy: {valid_acc}", flush = True)


def train(epochs, lrate, batch_size, qd, name):
	train_data = qd.train
	train_labels = qd.train_nlabels

	validation_data = qd.validation
	validation_labels = qd.validation_nlabels 

	wshape = {"theta": 24}

	start_time = time.time()

	c0layer = tf.keras.layers.Dense(8)
	qlayer = qml.qnn.KerasLayer(qcircuit, wshape, output_dim=1)
	clayer = tf.keras.layers.Dense(1, activation="sigmoid")
	model = tf.keras.models.Sequential([qlayer])

	opt = tf.keras.optimizers.Adam(learning_rate = lrate)
	model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy())
	history = model.fit(train_data, train_labels, epochs = epochs, shuffle = True, validation_data = (validation_data, validation_labels), verbose = True, batch_size = batch_size, callbacks = [CustomCallback(qd)])

	end_time = time.time()

	accuracy_train = get_accuracy(model, qd.train, qd.train_nlabels)
	accuracy_valid = get_accuracy(model, qd.validation, qd.validation_nlabels)


	if not (os.path.isdir("vqctf/out")):
		os.mkdir("vqctf/out")
	
	## Logging time!
	f = open("vqctf/out/log", "a")
	f.write("VQC " + name + "\n")
	f.write("Autoencoder: " + qd.model + "\n")
	f.write(circuit_desc + "\n")
	f.write(f"epochs/lrate/bsize: {epochs} / {lrate} / {batch_size}\n")
	f.write(f"train/valid {qd.ntrain} / {qd.nvalid}\n")
	f.write("Elapsed time: " + str(end_time - start_time) + "s " + str((end_time - start_time)/3600) + "h\n")
	f.write("Train accuracy: " + str(accuracy_train) + "\n")
	f.write("Valid accuracy: " + str(accuracy_valid) + "\n")
	f.write("\n\n")
	f.close();

	os.mkdir("vqctf/out/" + name)
	model.save_weights("vqctf/out/" + name + "/" + name)

	errs = history.history['loss'];
	plt.figure();
	plt.plot(range(len(errs)), errs)
	plt.title("Loss for " + name)
	plt.xlabel("Epochs")
	plt.savefig("vqctf/out/plot-" + name + ".png");

	return model, history
