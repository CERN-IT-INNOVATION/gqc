import numpy as np
import pennylane as qml
from pennylane.optimize import AdagradOptimizer

from vqctf.circuit import *
from vqctf.train import *
from qdata import qdata

# Number of events for signal/background. Total will be doubled!
ntrain = 2000
nvalid = 100
ntest = 0

qd = qdata("tf", ntrain, nvalid, ntest, shuffle = True)

epochs = 30
learning_rate = 0.001
batch_size = 50

name = "NM1"

model, hist = train(epochs, learning_rate, batch_size, qd, name)

qd = qdata("tf")
valid = qd.get_kfold_validation()

encoded = []

for i in range(len(valid)):
	sample = valid[i]
	encoded.append(np.array(model.predict(sample)))

encoded = np.array(encoded)

np.save("vqctf/out/" + name, encoded)


#acc = accuracy_score(qd.test_nlabels, test(theta, qd.test))

