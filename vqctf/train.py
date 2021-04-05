import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from vqcroc import *

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


def create_model(spec):
	params = 0
	for i in range(1, len(spec)):
		is_vf = False
		name = spec[i][0]
		if (name == "2local"):
			is_vf = True

		if (is_vf):
			params += spec[i][2] - spec[i][1]
	
	wshape = {"theta": params}
	qlayer = qml.qnn.KerasLayer(get_circuit(spec), wshape, output_dim=1)
	model = tf.keras.models.Sequential([qlayer])
	return model


def train(epochs, lrate, batch_size, spec, ntrain, encoder, name):

	nvalid = 100	
	qd = qdata(encoder, ntrain, nvalid)
	
	train_data = qd.train
	train_labels = qd.train_nlabels

	validation_data = qd.validation
	validation_labels = qd.validation_nlabels 

	model = create_model(spec)
	opt = tf.keras.optimizers.Adam(learning_rate = lrate)
	model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy())

	start_time = time.time()
	history = model.fit(train_data, train_labels, epochs = epochs, shuffle = True, validation_data = (validation_data, validation_labels), verbose = True, batch_size = batch_size)
	end_time = time.time()

	if not (os.path.isdir("vqctf/out")):
		os.mkdir("vqctf/out")
	dirname = "vqctf/out/" + name
	if not (os.path.isdir(dirname)):
		os.mkdir(dirname)

	print("Saving model...", flush = True)

	weights = np.array(model.get_weights())
	tloss = np.array(history.history['loss'])
	vloss = np.array(history.history['val_loss'])

	np.save(dirname + "/weights.npy", weights)
	np.save(dirname + "/tloss.npy", tloss)
	np.save(dirname + "/vloss.npy", vloss)
	np.save(dirname + "/features.npy", np.array([encoder]))
	np.save(dirname + "/spec.npy", np.array(spec, dtype = object))

	plt.figure();
	plt.plot(range(len(tloss)), tloss)
	plt.plot(range(len(vloss)), vloss)
	plt.title(name)
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	plt.savefig("vqctf/out/" + name + "/loss.pdf");
	plt.close()

	print("Computing predictions...", flush = True)

	qd = qdata(encoder)
	valid = qd.get_kfold_validation()
	encoded = []
	for i in range(len(valid)):
		sample = valid[i]
		encoded.append(np.array(model.predict(sample)))
		print(f"{i+1}/{len(valid)}",flush = True)
	encoded = np.array(encoded)

	print("Computing AUCs and saving...", flush = True)

	np.save(dirname + "/encoded.npy", encoded)
	info = get_info(name)
	auc_valid = info[2]
	auc_std = info[3]
	get_plot({name : info}, ntrain, folder = '/' + name)


	## Logging time!
	f = open("vqctf/out/log", "a")
	f.write("VQC " + name + "\n")
	f.write(f"epochs/lrate/bsize: {epochs} / {lrate} / {batch_size}\n")
	f.write(f"train/valid {ntrain} / {nvalid}\n")
	f.write("Elapsed time: " + str(end_time - start_time) + "s " + str((end_time - start_time)/3600) + "h\n")
	f.write("Valid AUC: " + str(auc_valid) + "+/-" + str(auc_std) + "\n")
	f.write("\n\n")
	f.close();

	return model



