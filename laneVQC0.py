import numpy as np
import pennylane as qml
from pennylane.optimize import AdagradOptimizer

from vqc.circuit import *
from vqc.train import *
from qdata import qdata

# Number of events for signal/background. Total will be doubled!
ntrain = 200
nvalid = 50
ntest = 0

qd = qdata("tf", ntrain, nvalid, ntest, shuffle = True)

epochs = 20
learning_rate = 0.0025
batch_size = 50

train(epochs, learning_rate, batch_size, qd, "test")

#acc = accuracy_score(qd.test_nlabels, test(theta, qd.test))

