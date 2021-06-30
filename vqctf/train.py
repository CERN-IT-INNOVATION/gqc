import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
	params = 0 # Keeps track of the number of parameters used in the variational forms (needed).
	spec_qc = [spec[0]] #
	layer_seq = []
	for i in range(1, len(spec)):
		is_vf = False
		is_qc = True
		name = spec[i][0]

		if (name == "elu"):
			is_qc = False
			layer_seq.append(layers.Dense(spec[i][1], activation = 'elu'))

		if (name == "2local"):
			is_vf = True
		elif (name == "tree"):
			is_vf = True
		elif (name == "step"):
			is_vf = True
		
		if (is_vf):
			params += spec[i][2] - spec[i][1]
		if (is_qc):
			spec_qc.append(spec[i])
	
	wshape = {"theta": params}
	# When defining the quantum layer, get_circuit is called with a "quantum spec".
	# This is spec without the details of the classical layers in the model.
	qlayer = qml.qnn.KerasLayer(get_circuit(spec_qc), wshape, output_dim=1)
	layer_seq.append(qlayer)
	model = tf.keras.models.Sequential(layer_seq)
	return model


# The names of all the variables are self-explanatory except spec.
# Spec is an array that defines the architecture of the model:
# 	First element can be either:
# 		- The number of qubits.
# 		- An array [number of qubits, observable used] 
#	The remaining elements specify the elements of the architecture in sequential order.
#	See create_model()-->circuit/get_circuit() for details.
def train(epochs, lrate, batch_size, spec, ntrain, encoder, name):

	# Get the data and define some variables.
	nvalid = 100	
	qd = qdata(encoder, ntrain, nvalid, use_complex = True)
	
	train_data = qd.train
	train_labels = qd.train_nlabels

	validation_data = qd.validation
	validation_labels = qd.validation_nlabels 

	# Create the keras model and set things up for training.
	model = create_model(spec)
	opt = tf.keras.optimizers.Adam(learning_rate = lrate)
	model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy())

	earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)

	# Train! (And keep track of time).
	start_time = time.time()
	history = model.fit(train_data, train_labels, epochs = epochs, shuffle = True, validation_data = (validation_data, validation_labels), verbose = True, batch_size = batch_size, callbacks = [earlystop])
	end_time = time.time()

	# Process the output.
	epused = len(history.history['loss'])

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

	qd = qdata(encoder, use_complex = True)
	valid = qd.get_kfold_validation()
	test = qd.get_kfold_test()
	encoded_val = []
	encoded_test = []
	for i in range(len(valid)):
		sample_val = valid[i]
		sample_test = test[i]
		encoded_val.append(np.array(model.predict(sample_val)))
		encoded_test.append(np.array(model.predict(sample_test)))
		print(f"{i+1}/{len(valid)}",flush = True)
	encoded_val = np.array(encoded_val)
	encoded_test = np.array(encoded_test)

	print("Computing AUCs and saving...", flush = True)

	np.save(dirname + "/encoded_val.npy", encoded_val)
	np.save(dirname + "/encoded_test.npy", encoded_test)
	info_val = get_info(name + "/encoded_val.npy")
	info_test = get_info(name + "/encoded_test.npy")
	auc_valid = info_val[2]
	auc_valid_std = info_val[3]
	auc_test = info_test[2]
	auc_test_std = info_test[3]
	get_plot({name + "_valid": info_val}, ntrain, folder = '/' + name)
#	get_plot({name + "_test": info_test}, ntrain, folder = '/' + name)


	## Logging time!
	f = open("vqctf/out/log", "a")
	f.write("VQC " + name + "\n")
	f.write(f"epochs/lrate/bsize: {epused} / {lrate} / {batch_size}\n")
	f.write(f"train/valid {ntrain} / {nvalid}\n")
	f.write("Elapsed time: " + str(end_time - start_time) + "s " + str((end_time - start_time)/3600) + "h\n")
	f.write("Valid AUC: " + str(auc_valid) + "+/-" + str(auc_valid_std) + "\n")
#	f.write("Test AUC: " + str(auc_test) + "+/-" + str(auc_test_std) + "\n")
	f.write("\n\n")
	f.close();

	return model



