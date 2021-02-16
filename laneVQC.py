import numpy as np
import pennylane as qml
from pennylane.optimize import AdagradOptimizer

from vqctf.circuit import *
from vqctf.train import *
from qdata import qdata

# Number of events for signal/background. Total will be doubled!
ntrain = 1500
nvalid = 500
ntest = 0

qd = qdata("tf", ntrain, nvalid, ntest, shuffle = True)

epochs = 150
learning_rate = 0.025
batch_size = 500

model, hist = train(epochs, learning_rate, batch_size, qd, "TF1-Big-1")

#acc = accuracy_score(qd.test_nlabels, test(theta, qd.test))

