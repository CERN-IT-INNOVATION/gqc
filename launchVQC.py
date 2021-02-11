from qiskit.ml.datasets import *
from qiskit.aqua.components.optimizers import CRS, TNC, ISRES # Classical optimizer (any would do).
from qiskit.circuit.library.n_local.two_local import TwoLocal # Generates the variational circuit.
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.

import time
from datetime import datetime
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC

import vqc.zzcool as fmap
import qdata
from aeTF.encode import model
from aeTF.encode import encode

nqubits = 3
nfeatures = 3
ntrain = 80 # Number of events for signal/background. Total will be doubled!
ntest = 60
nreps = 2
nrepsvf = 1

start_time = time.time()

train_data = qdata.train_dict
train_data['s'] = train_data['s'][range(ntrain),]
train_data['b'] = train_data['b'][range(ntrain),]
test_data = qdata.test_dict
test_data['s'] = test_data['s'][range(ntest),]
test_data['b'] = test_data['b'][range(ntest),]

train_data = encode(train_data)
test_data = encode(test_data)

train_data['s'] = train_data['s'][range(ntrain),[1,3,7]]
train_data['b'] = train_data['b'][range(ntrain),[1,3,7]]

test_data['s'] = test_data['s'][range(ntest),[1,3,7]]
test_data['b'] = test_data['b'][range(ntest),[1,3,7]]
var_form = TwoLocal(nqubits, 'ry', 'cx', 'linear', reps = nrepsvf, insert_barriers=False)
feature_map = fmap.get_circuit(nqubits, nfeatures, reps = nreps)

vqc = VQC(ISRES(), feature_map, var_form, train_data, test_data)
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)
result = vqc.run(quantum_instance)

out_s = vqc.predict(train_data['s'])[1]
out_b = vqc.predict(train_data['b'])[1]

good_s = sum(out_s == vqc.class_to_label['s'])
good_b = sum(out_b == vqc.class_to_label['b'])
train_acc = (good_s + good_b) / (len(out_s) + len(out_b))


print("testing success ratio: {}".format(result['testing_accuracy']))
print("training success ratio: {}".format(train_acc))

end_time = time.time()
print(f"Execution Time {end_time-start_time} s")


## Logging time!

f = open("vqc/log.txt", "a")
f.write("VQC LOG " + str(datetime.now()) + "\n")
f.write("Feature map: " + fmap.description + " (reps = " + str(nreps) + ")\n")

###### ADJUST THIS!! ######
f.write("2-local var form (reps = " + str(nrepsvf) + ")\n")

f.write("Autoencoder: " + model + "\n")
f.write("Qubits: " + str(nqubits) + "\n")
f.write("Used features: " + str(nfeatures) + "\n")
f.write("Training samples: " + str(ntrain) + " Testing samples: " + str(ntest) + "\n")
f.write("Elapsed time: " + str(end_time - start_time) + "s " + str((end_time - start_time)/3600) + "h\n")
f.write("Training accuracy: " + str(train_acc) + "\n")
f.write("Accuracy: " + str(result['testing_accuracy']) + "\n")
f.write("\n\n")
f.close();
