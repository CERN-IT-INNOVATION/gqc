from qiskit.ml.datasets import *
from qiskit.aqua.components.optimizers import CRS, TNC, ISRES # Classical optimizer (any would do).
from qiskit.circuit.library.n_local.two_local import TwoLocal # Generates the variational circuit.
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.

import time
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC

import qdata
from aePyTorch.encode import encode

start_time = time.time()
feature_dim = 4
var_form = TwoLocal(feature_dim, 'ry', 'cx', 'linear', reps = 1, insert_barriers=False)
feature_map = RawFeatureVector(2**feature_dim)

vqc = VQC(ISRES(), feature_map, var_form, encode(qdata.train_dict), encode(qdata.validation_dict))

print("hey there!")

seed = 10598
backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

result = vqc.run(quantum_instance)

print("testing success ratio: {}".format(result['testing_accuracy']))
end_time = time.time()
print(f"Execution Time {end_time-start_time} s")
