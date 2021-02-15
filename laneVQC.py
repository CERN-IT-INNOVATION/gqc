import numpy as np
import pennylane as qml
from pennylane.optimize import AdagradOptimizer

from vqctf.circuit import *
from vqctf.train import *
from qdata import qdata

# Number of events for signal/background. Total will be doubled!
ntrain = 2000
nvalid = 2000
ntest = 0

qd = qdata("tf", ntrain, nvalid, ntest, shuffle = True)

epochs = 200
learning_rate = 0.025
batch_size = 600

model, hist = train(epochs, learning_rate, batch_size, qd, "TF1-Big-1")

#acc = accuracy_score(qd.test_nlabels, test(theta, qd.test))

