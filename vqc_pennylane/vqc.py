import pennylane as pnl
import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import feature_maps as fm
import variational_forms as vf


class VQC:
    """
    Variational quantum circuit, implemented using the pennylane python
    package. This is a trainable quantum circuit. It is composed of a feature
    maps and a variational form, which are implemented in their eponymous
    files in the same directory.
    """
    def __init__(self, nqubits, nfeatures):
        """
        @nqubits :: Number of qubits the circuit should be made of.
        """
        self._check_compatibility(nqubits, nfeatures)
        self._nfeatures = nfeatures
        self._nqubits = nqubits
        self.dev = pnl.device("default.qubit", wires=nqubits)

        self.qcircuit = pnl.qnode(self.dev)(self.qcircuit)

    def _qcircuit(self, inputs, weights):
        """
        The quantum circuit builder.
        @inputs  :: The inputs taken by the feature maps.
        @weights :: The weights of the variational forms used.

        returns :: Measurement of the first qubit of the quantum circuit.
        """
        for feature_idx in range(self._nfeatures):
            start_feature_idx = feature_idx
            end_feature_idx = feature_idx + self._nqubits
            fm.zzfm(self.nqubits, inputs[start_feature_idx:end_feature_idx])
            vf.two_local()

            feature_idx += self._nqubits

        return pnl.expval(pnl.Hermitian(y, wires=[0]))

    @property
    def nqubits(self):
        return self._nqubits

    @property
    def nfeatures(self):
        return self._nfeatures

    @property
    def qcircuit(self):
        return self._qcircuit

    @staticmethod
    def _check_compatibility(nqubits, nfeatures):
        """
        Checks if the number of features in the dataset is divisible by
        the number of qubits.
        @nqubits   :: Number of qubits assigned to the vqc.
        @nfeatures :: Number of features to process by the vqc.
        """
        if nqubits % nfeatures != 0:
            raise ValueError("The number of features is not divisible by "
                             "the number of qubits you assigned!")
