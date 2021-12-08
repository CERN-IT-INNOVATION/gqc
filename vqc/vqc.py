"""
VQC class implemented with qiskit and qiskit machine learning.
"""
import numpy as np

from typing import cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers.neural_network_classifier \
    import NeuralNetworkClassifier


class VQC(NeuralNetworkClassifier):
    def __init__(self, num_qubits=None, feature_map=None, ansatz=None,
                 input_dim=None, warm_start=False, quantum_instance=None,
                 initial_point=None, loss=None, optimizer=None):
        """
        @num_qubits    :: The number of qubits for the underlying CircuitQNN.
                          If None, derive from feature_map or ansatz.
                          If neither of those is given, raise exception.
        @feature_map   :: The feature map for underlying CircuitQNN.
                          If None, use ZZFeatureMap.
        @ansatz        :: The ansatz for the underlying CircuitQNN.
                          If None, use TwoLocal.
        @warm_start    :: Use weights from previous fit to start next fit.
        @initial_point :: Initial point for the optimizer to start from.
        """

        # Perform initialisation checks on variables.
        if num_qubits is None and feature_map is None and ansatz is None:
            raise AttributeError("Input num_qubits, feature_map, or ansatz!")

        self._input_dim = input_dim
        self._num_qubits = None
        self._vforms = []

        self._configure_vqc_circuit(num_qubits, feature_map, ansatz)

        # Construct the quantum circuit.
        self._circuit = QuantumCircuit(self._num_qubits)
        for feature_map, ansatz in zip(self._vforms[0::2], self._vforms[1::2]):
            print(feature_map)
            self._circuit.compose(feature_map, inplace=True)
            self._circuit.compose(ansatz, inplace=True)

        print(self._circuit)
        # Construct the circuit of the QNN.
        neural_network = CircuitQNN(
            self._circuit,
            self._feature_map.parameters,
            self._ansatz.parameters,
            interpret=self._get_interpret(2),
            output_shape=2,
            quantum_instance=quantum_instance,
            input_gradients=True,
        )
        super().__init__(
            neural_network=neural_network,
            loss=loss,
            one_hot=False,
            optimizer=optimizer,
            warm_start=warm_start,
            initial_point=initial_point,
        )

    def _configure_vqc_circuit(self, num_qubits, feature_map, ansatz):
        if num_qubits:
            self._check_input_compatibility(num_qubits)
            self._config_with_num_qubits(num_qubits, feature_map, ansatz)
        else:
            if feature_map:
                self._config_with_feature_map(feature_map, ansatz)
            else:
                self._config_with_ansatz(ansatz)

    def _config_with_num_qubits(self, num_qubits, feature_map, ansatz):
        self._num_qubits = num_qubits

        for form_nb in range(int(self._input_dim/num_qubits)):
            self._set_numq_feature_map(feature_map)
            self._set_numq_ansatz(ansatz)

    def _set_numq_feature_map(self, feature_map):
        if feature_map:
            if feature_map.num_qubits != self._num_qubits:
                raise AttributeError("Incompat num_qubits and feature_map!")
            self._vforms.append(feature_map)
        else:
            self._vforms.append(
                ZZFeatureMap(self._num_qubits, 1, "linear"))

    def _set_numq_ansatz(self, ansatz):
        if ansatz:
            if ansatz.num_qubits != self._num_qubits:
                raise AttributeError("Incompatible num_qubits and ansatz!")
            self._vforms.append(ansatz)
        else:
            self._vforms.append(
                TwoLocal(self._num_qubits, 'ry', 'cx', 'linear', 1))

    def _config_with_feature_map(self, feature_map, ansatz):
        self._check_input_compatibility(feature_map.num_qubits)
        self._num_qubits = feature_map.num_qubits
        if ansatz:
            if feature_map.num_qubits != ansatz.num_qubits:
                raise AttributeError("Incompatible feature_map and ansatz!")
            self._vforms.append(feature_map)
            self._vforms.append(ansatz)
        else:
            self._vforms.append(feature_map)
            self._vforms.append(
                TwoLocal(self._num_qubits, 'ry', 'cx', 'linear', 1))

    def _config_with_ansatz(self, ansatz):
        self._check_input_compatibility(ansatz.num_qubits)
        self._num_qubits = ansatz.num_qubits
        self._vforms.append(
            ZZFeatureMap(self._num_qubits, 1, "linear"))
        self._vforms.append(ansatz)

    def _check_input_compatibility(self, num_qubits):
        if self._input_dim % num_qubits != 0:
            raise AttributeError("The dimensions of your input should be "
                                 "divisible by the number of qubits.")

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the used feature map."""
        return self._feature_map

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the used ansatz."""
        return self._ansatz

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return self.circuit.num_qubits

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to data matrix X and targets y.

        @X :: The input data.
        @y :: The target values.

        returns: self: returns a trained classifier.
        """
        num_classes = len(np.unique(y, axis=0))
        cast(CircuitQNN, self._neural_network).set_interpret(
            self._get_interpret(num_classes), num_classes)
        return super().fit(X, y)

    def _get_interpret(self, num_classes):
        def parity(x, num_classes=num_classes):
            return "{:b}".format(x).count("1") % num_classes

        return parity
