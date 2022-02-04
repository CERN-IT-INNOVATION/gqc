import pennylane as pnl
import numpy as np

from . import feature_maps as fm
from . import variational_forms as vf
from .terminal_colors import tcols


class VQC:
    """
    Variational quantum circuit, implemented using the pennylane python
    package. This is a trainable quantum circuit. It is composed of a feature
    map and a variational form, which are implemented in their eponymous
    files in the same directory.
    """
    def __init__(self, nqubits, nfeatures, fmap="zzfm", vform="two_local",
        vform_repeats=4):
        """
        @nqubits   :: Number of qubits the circuit should be made of.
        @nfeatures :: Number of features in the training data set.
        @fmap      :: String name of the feature map to use.
        @vform     :: String name of the variational form to use.
        """
        self._nsubforms = self._check_compatibility(nqubits, nfeatures)
        self._nfeatures = nfeatures
        self._nqubits = nqubits
        self._vform_repeats = vform_repeats
        self._nweights = vf.vforms_weights(vform, vform_repeats, nqubits)

        self._device = pnl.device("default.qubit", wires=nqubits)
        self._circuit = pnl.qnode(self._device)(self._qcircuit)

    def _qcircuit(self, inputs, weights):
        """
        The quantum circuit builder.
        @inputs  :: The inputs taken by the feature maps.
        @weights :: The weights of the variational forms used.

        returns :: Measurement of the first qubit of the quantum circuit.
        """
        for subform_nb in range(self._nsubforms-1):
            start_feature = subform_nb*self._nqubits
            start_weights = self._nweights*subform_nb
            end_feature = self._nqubits*(subform_nb + 1)
            end_weights = self._nweights*(subform_nb + 1)

            fm.zzfm(self._nqubits, inputs[start_feature:end_feature])
            vf.two_local(self._nqubits, weights[start_weights:end_weights],
                         repeats=self._vform_repeats, entanglement="linear")

        y = [[1], [0]] * np.conj([[1], [0]]).T
        return pnl.expval(pnl.Hermitian(y, wires=[0]))

    @property
    def nqubits(self):
        return self._nqubits

    @property
    def nfeatures(self):
        return self._nfeatures

    @property
    def circuit(self):
        return self._circuit

    @property
    def subforms(self):
        return self._subforms

    @property
    def nweights(self):
        return self._nweights

    def draw(self):
        """
        Draws the circuit using dummy parameters.
        Parameterless implementation is not yet available in pennylane,
        and it seems not feasible either by the way pennylane is constructed.
        """
        drawing = pnl.draw(self._circuit)
        print(tcols.OKGREEN)
        print(drawing([0]*int(self._nfeatures),
                      [0]*int(self._nweights*self._nsubforms)))
        print(tcols.ENDC)

    @staticmethod
    def _check_compatibility(nqubits, nfeatures):
        """
        Checks if the number of features in the dataset is divisible by
        the number of qubits.
        @nqubits   :: Number of qubits assigned to the vqc.
        @nfeatures :: Number of features to process by the vqc.
        """
        if nfeatures % nqubits != 0:
            raise ValueError("The number of features is not divisible by "
                             "the number of qubits you assigned!")

        return int(nfeatures/nqubits)
