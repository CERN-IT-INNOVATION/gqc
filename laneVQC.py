import numpy as np
import pennylane as qml
from pennylane.optimize import AdamOptimizer

from vqc.circuit import *
from qdata import qdata

import time
from datetime import datetime

# Number of events for signal/background. Total will be doubled!
ntrain = 1000
nvalid = 200
ntest = 100

start_time = time.time()

qd = qdata("tf", ntrain, nvalid, ntest, False)

train_data = qd.train
train_labels = qd.train_nlabels
validation_data = qd.validation
validation_labels = qd.validation_nlabels
test_data = qd.test
test_labels = qd.test_nlabels

epochs = 100
learning_rate = 0.1
batch_size = 10

theta = np.random.uniform(size=24)

print("Running!")

opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

accuracy_train = 0
accuracy_test = 0

for it in range(epochs):
	for data_batch, target_batch in iterate_minibatches(train_data, train_labels, batch_size = batch_size):
		theta = opt.step(lambda v: cost(v, data_batch, target_batch), theta)


	predicted_train, fidel_train = test(theta, train_data)
	accuracy_train = accuracy_score(train_labels, predicted_train)
	loss = cost(theta, train_data, train_labels)

	predicted_test, fidel_test = test(theta, test_data)
	accuracy_test = accuracy_score(test_labels, predicted_test)
	res = [it + 1, loss, accuracy_train, accuracy_test]
	print(
		"Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(*res)
	)


end_time = time.time()
print(theta)

## Logging time!

f = open("vqc/log.txt", "a")
f.write("VQC LOG " + str(datetime.now()) + "\n")
f.write("Feature map: " + fmap.description + "\n")
f.write("Variational form: " + vform.description + "\n")
#f.write("Autoencoder: " + model + "\n")
f.write("Qubits: " + str(nqubits) + "\n")
f.write("epochs/lrate/bsize: " + str(epochs) + "/" + str(learning_rate) + "/" + str(batch_size) + "\n")
f.write("train/valid/test: " + str(ntrain) + "/" + str(nvalid) + "/" + str(ntest) +  "\n")
f.write("Elapsed time: " + str(end_time - start_time) + "s " + str((end_time - start_time)/3600) + "h\n")
f.write("Train accuracy: " + str(accuracy_train) + "\n")
f.write("Test accuracy: " + str(accuracy_test) + "\n")
f.write("\n\n")
f.close();
