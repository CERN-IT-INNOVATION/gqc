import pennylane as qml
import numpy as np
import pickle

from pennylane.optimize import AdamOptimizer
from sklearn.metrics import log_loss
from vqc.circuit import *

import time
from datetime import datetime

def cost(theta, data, labels):
	loss = 0.0
	for i in range(len(data)):
		f = qcircuit(theta, data[i])
		if (labels[i] == 0):
			loss += (1-f)**2
		else:
			loss += f**2;
	return loss / len(data)


def accuracy_score(label_true, label_pred):
	score = label_true == label_pred
	return score.sum() / len(label_true)


def iterate_minibatches(inputs, targets, batch_size):
	for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
		idxs = slice(start_idx, start_idx + batch_size)
		yield inputs[idxs], targets[idxs]


def train(epochs, lrate, batch_size, qd, name = "default"):
	
	train_data = qd.train
	train_labels = qd.train_nlabels
	validation_data = qd.validation
	validation_labels = qd.validation_nlabels
	
	opt = AdamOptimizer(lrate);

	accuracy_train = 0
	accuracy_valid = 0
	
	theta = np.random.uniform(size=25)
	
	print("Running!", flush = True)
	
	start_time = time.time()

	for it in range(epochs):
		for data_batch, target_batch in iterate_minibatches(train_data, train_labels, batch_size = batch_size):
			theta = opt.step(lambda v: cost(v, data_batch, target_batch), theta)
			
		predicted_train = test(theta, train_data)
		accuracy_train = accuracy_score(train_labels, predicted_train)
		loss = cost(theta, train_data, train_labels)
		
		predicted_valid = test(theta, validation_data)
		accuracy_valid = accuracy_score(validation_labels, predicted_valid)
		res = [it + 1, loss, accuracy_train, accuracy_valid]
		print(
			"Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Valid accuracy: {:3f}".format(*res), flush = True
		)


	end_time = time.time()
	
	
	## Logging time!
	
	f = open("vqc/out/log", "a")
	f.write("VQC LOG " + name + "\n")
	f.write("Autoencoder: " + qd.model + "\n")
	f.write(circuit_desc + "\n")
	f.write(f"epochs/lrate/bsize: {str(epochs)} / {lrate} / {batch_size}\n")
	f.write(f"train/valid: {qd.ntrain} / {qd.nvalid}\n")
	f.write("Elapsed time: " + str(end_time - start_time) + "s " + str((end_time - start_time)/3600) + "h\n")
	f.write("Train accuracy: " + str(accuracy_train) + "\n")
	f.write("Test accuracy: " + str(accuracy_valid) + "\n")
	f.write("\n\n")
	f.close();

	filename = "vqc/out/" + name;
	save = {"theta": theta, "circuit": circuit_desc, "epochs": epochs, "lrate": lrate, "bsize": batch_size}
	
	f = open(filename, "wb")
	pickle.dump(theta, f)
	f.close()
	
	return theta;

