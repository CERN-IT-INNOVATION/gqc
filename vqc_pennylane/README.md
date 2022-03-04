# (Hybrid) VQC model in pennylane

* In `util.noisy_simulation` no need to seperately include "transpile\_args" since `pennylane` takes care of the setting using the `qiskit.transpile` signature. The remaining keywords that are provided through `config` are assumed to be `run_args` [source][https://github.com/PennyLaneAI/pennylane-qiskit/blob/master/pennylane\_qiskit/qiskit\_device.py#L180]. *TODO*: test if qiskit complains for some of the `run_args` passed by `pennylane`.
* *TODO*: If we want to use `qsvm.util.save_circuit_physical_layout` we need to translate `pnl.qnode` circuit to QASm and then from that format rebuilt qiskit.QuantumCircuit.

Note: The type of the model is specified by --hybrid <hybrid/non-hybrd> and --class_weight <weight of the VQC classification branch>.
