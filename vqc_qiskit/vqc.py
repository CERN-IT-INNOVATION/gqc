# VQC class implemented with qiskit and qiskit machine learning.
# If the number of features is larger than the number of available qubits,
# the circuit is assembled in such a way to use data reouploading.
# This is highly experimental and certain hacks are employed to make this
# possible.

from typing import cast
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers.neural_network_classifier \
    import NeuralNetworkClassifier

from .zzfeaturemap import ZZFeatureMap


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
        self._check_input_arguments(num_qubits, feature_map, ansatz)
        self._roman_letters = 'abcdefghijklmnopqrstuvwxyz'
        self._greek_letters = 'αβγδεζηθικλμνξοπρστυφχψω'

        self._input_dim = input_dim
        self._num_qubits = None

        self._input_params = []
        self._weight_params = []

        vforms = self._configure_vqc_circuit(num_qubits, feature_map, ansatz)

        # Construct the quantum circuit.
        self._circuit = QuantumCircuit(self._num_qubits)
        for feature_map, ansatz in zip(vforms[0::2], vforms[1::2]):
            self._input_params += feature_map.parameters
            self._weight_params += ansatz.parameters
            self._circuit.compose(feature_map, inplace=True)
            self._circuit.compose(ansatz, inplace=True)

        # Construct the circuit of the QNN.
        neural_network = CircuitQNN(
            self._circuit,
            self._input_params,
            self._weight_params,
            interpret=self._get_interpret(2),
            output_shape=2,
            quantum_instance=quantum_instance,
            input_gradients=False,
        )

        super().__init__(
            neural_network=neural_network,
            loss=loss,
            one_hot=True,
            optimizer=optimizer,
            warm_start=warm_start,
            initial_point=initial_point,
        )

    def _configure_vqc_circuit(self, num_qubits, feature_map, ansatz):
        """
        Father method for all the configure methods below.
        @num_qubits  :: Int of the number of qubits to be used.
        @feature_map :: The feature map object to be used. Can be none.
        @ansatz      :: The ansatz object to be used. Can be none.

        returns :: List of the variational forms used to construct the
                   vqc qiskit circuit.
        """
        if num_qubits:
            return self._config_with_num_qubits(num_qubits, feature_map, ansatz)
        else:
            if feature_map:
                return self._config_with_feature_map(feature_map, ansatz)
            else:
                return self._config_with_ansatz(ansatz)

    def _config_with_num_qubits(self, num_qubits, feature_map, ansatz):
        """
        Configure the vqc given the number of qubits. If the feature map
        and ansatz are none, choose default ones.

        @num_qubits  :: Int of the number of qubits to be used.
        @feature_map :: The feature map object to be used. Can be none.
        @ansatz      :: The ansatz object to be used. Can be none.

        returns :: List of the variational forms used to construct the
                   vqc qiskit circuit.
        """
        self._check_input_compatibility(num_qubits)
        self._num_qubits = num_qubits
        vforms = []

        for form_nb in range(int(self._input_dim/num_qubits)):
            vforms.append(self._set_feature_map(feature_map))
            vforms.append(self._set_ansatz(ansatz))

        return vforms

    def _config_with_feature_map(self, feature_map, ansatz):
        """
        Configure the vqc given no number of qubits but a feature map instead.
        The number of qubits are taken from the feature map attributes.
        The ansatz, if not given, is defaulted to TwoLocal.
        @feature_map :: The feature map object to be used. Can be none.
        @ansatz      :: The ansatz object to be used. Can be none.

        returns :: List of the variational forms used to construct the
                   vqc qiskit circuit.
        """
        self._check_input_compatibility(feature_map.num_qubits)
        self._num_qubits = feature_map.num_qubits
        vforms = []

        for form_nb in range(int(self._input_dim/num_qubits)):
            vforms.append(self._set_feature_map(feature_map))
            vforms.append(self._set_ansatz(ansatz))

        return vforms

    def _config_with_ansatz(self, ansatz):
        """
        Configure the vqc given no number of qubits to be used or feature map.
        Then, an ansatz must be given. The number of qubits are used from the
        ansatz. The feature map defaults to ZZFeatureMap.
        @ansatz      :: The ansatz object to be used. Can be none.

        returns :: List of the variational forms used to construct the
                   vqc qiskit circuit.
        """
        self._check_input_compatibility(ansatz.num_qubits)
        self._num_qubits = ansatz.num_qubits
        vforms = []

        for form_nb in range(int(self._input_dim/num_qubits)):
            vforms.append(self._set_feature_map(feature_map))
            vforms.append(self._set_ansatz(ansatz))

        return vforms

    def _set_feature_map(self, feature_map):
        """
        Checks if the feature map exists. If it does, check consistency with
        the available number of qubits in the circuit and return it.
        Otherwise, if the feature map does not exist, default to ZZFeatureMap.
        @feature_map :: The feature map object to be used. Can be none.

        returns :: Qiskit circuit object of a feature map.
        """
        if feature_map:
            if feature_map.num_qubits != self._num_qubits:
                raise AttributeError("Incompat num_qubits and feature_map!")
            return feature_map

        param_prefix = self._roman_letters[-1]
        self._roman_letters = self._roman_letters[:-1]

        return ZZFeatureMap(self._num_qubits, 1, "linear",
                            parameter_prefix=param_prefix)

    def _set_ansatz(self, ansatz):
        """
        Check if the ansatz exists. If it does, check consistency with the
        available number of qubits and return it.
        Otherwise, if the ansatz does not exist, default to TwoLocal.
        @ansatz :: The ansatz object to be used. Can be none.

        returns :: Qiskit circuit object of an ansatz.
        """
        if ansatz:
            if ansatz.num_qubits != self._num_qubits:
                raise AttributeError("Incompatible num_qubits and ansatz!")
            return ansatz

        param_prefix = self._greek_letters[-1]
        self._greek_letters = self._greek_letters[:-1]

        return TwoLocal(self._num_qubits, 'ry', 'cx', 'linear', 1,
                        parameter_prefix=param_prefix)

    @staticmethod
    def _check_input_arguments(num_qubits, feature_map, ansatz):
        """
        Check if either a number of qubits, a feature map, or an ansatz was
        given by the user.
        """
        if num_qubits is None and feature_map is None and ansatz is None:
            raise AttributeError("Give num_qubits, feature_map, or ansatz!")

    def _check_input_compatibility(self, num_qubits):
        """
        Check if the input dimension is compatible with the number of qubits
        that are being used.
        @num_qubits :: Int of the number of qubits used in the vqc.
        """
        if self._input_dim % num_qubits != 0:
            raise AttributeError("The dimensions of your input should be "
                                 "divisible by the number of qubits.")

    def _encode_onehot(self, target):
        """
        Reshape the target that such that it follows onehot encoding.
        @target :: Numpy array with target data.
        """
        onehot_target = np.zeros((target.size, int(target.max()+1)))
        onehot_target[np.arange(target.size), target.astype(int)] = 1

        return onehot_target

    def _check_target_encoding_onehot(self, target):
        """
        This class is hardcoded to only work with onehot encoding.
        Thus, check if the provided target data is onehot encoded.
        @target :: Numpy array containing the target data.
        """
        num_classes = len(np.unique(target, axis=0))
        if num_classes == target.shape[0]:
            return 1
        return 0

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits

    @property
    def input_params(self):
        return self._input_params

    @property
    def weight_params(self):
        return self._weight_params

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to data matrix X and targets y.

        @X :: The input data.
        @y :: The target values.

        returns: self: returns a trained classifier.
        """
        if not self._check_target_encoding_onehot(y):
            y = self._encode_onehot(y)
        num_classes = len(np.unique(y, axis=0))
        cast(CircuitQNN, self._neural_network).set_interpret(
            self._get_interpret(num_classes), num_classes)

        return super().fit(X, y)

    def _get_interpret(self, num_classes):
        """
        In the current vqc implementation, either no measurements are done
        (the feature vectors are available), or every measurement is done,
        when hardware or a noisy simulation is used.
        Thus, we need to map the output of the circuit to something that
        we can compare with our target labels. Here's where this parity
        method comes it, which maps the even bits to background and the odd
        bits to signal, thus returning two numbers per data sample.
        These two numbers are the probability to be signal and background.
        This then requires the target to be one-hot encoded.
        @num_classes :: Int of the number of classes that are present in the
                        data, i.e., for one signal and one bkg this is 2.
        """
        def parity(x, num_classes=num_classes):
            return f"{x:b}".count("1") % num_classes

        return parity
