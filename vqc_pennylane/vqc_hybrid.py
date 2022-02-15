# Hybrid VQC.

import numpy as np
import torch
import torch.nn as nn
import pennylane as pnl

from autoencoders.ae_classifier import AE_classifier
from . import feature_maps as fm
from . import variational_forms as vf
from .terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need. Disabled to increase performance.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)


class VQCHybrid(AE_classifier):
    """
    Main skeleton for having a VQC classifier attached to
    latent space of an Autoencoder and for a hybrid encoder (classical NN) + VQC
    classifier. The latter is can be constructed from the former by "chopping"
    off the decoder branch of the Autoencoder.
        @qdevice :: String containing what kind of device to run the
                    quantum circuit on: simulation, or actual computer?
        @device  ::
        @hpars   :: Dictionary of the hyperparameters to configure the vqc.
    """

    def __init__(self, qdevice, device, hpars):
        super().__init__(device, hpars)
        del self.hp["class_layers"]
        new_hp = {
            "nqubits": 4,
            "nfeatures": 16,
            "fmap": "zzfm",
            "vform": "two_local",
            "vform_repeats": 4,
        }
        self.hp.update(new_hp)
        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())

        self._qdevice = pnl.device(qdevice, wires=self.hp["nqubits"])
        self._layers = self._check_compatibility(
            self.hp["nqubits"], self.hp["nfeatures"]
        )

        self.epochs_no_improve = 0

        self._vqc_nweights = vf.vforms_weights(
            self.hp["vform"], self.hp["vform_repeats"], self.hp["nqubits"]
        )
        self._weight_shape = {"weights": (self._layers, self._vqc_nweights)}

        self._circuit = pnl.qnode(self._qdevice, interface="torch")(
            self.__construct_classifier
        )
        self.classifier = pnl.qnn.TorchLayer(self._circuit, self._weight_shape)

    def __construct_classifier(self, inputs, weights):
        """
        The quantum circuit builder, it overides the method of AE_classifier.
        The VQC will be used as the classifier branch of the hybrid network.

        @inputs  :: The inputs taken by the feature maps.
        @weights :: The weights of the variational forms used.

        returns :: Measurement of the first qubit of the quantum circuit.
        """
        for layer_nb in range(self._layers):
            start_feature = layer_nb * self.hp["nqubits"]
            end_feature = self.hp["nqubits"] * (layer_nb + 1)
            fm.zzfm(self.hp["nqubits"], inputs[start_feature:end_feature])
            vf.two_local(
                self.hp["nqubits"],
                weights[layer_nb],
                repeats=self.hp["vform_repeats"],
                entanglement="linear",
            )

        y = [[1], [0]] * np.conj([[1], [0]]).T
        return pnl.expval(pnl.Hermitian(y, wires=[0]))

    @property
    def nqubits(self):
        return self._hp["nqubits"]

    @property
    def nfeatures(self):
        return self._hp["nfeatures"]

    @property
    def circuit(self):
        return self._circuit

    @property
    def nweights(self):
        return self._vqc_nweights

    @staticmethod
    def _check_compatibility(nqubits, nfeatures):
        """
        Checks if the number of features in the dataset is divisible by
        the number of qubits.
        @nqubits   :: Number of qubits assigned to the vqc.
        @nfeatures :: Number of features to process by the vqc.
        """
        if nfeatures % nqubits != 0:
            raise ValueError(
                "The number of features is not divisible by "
                "the number of qubits you assigned!"
            )

        return int(nfeatures / nqubits)

    def train_model(self, train_loader, valid_loader, epochs, estopping_limit, outdir):
        """
        Train the classifier autoencoder.
        @train_loader :: Pytorch data loader with the training data.
        @valid_loader :: Pytorch data loader with the validation data.
        @epochs       :: The number of epochs to train for.
        @outdir       :: The output dir where to save the train results.
        """
        self.instantiate_adam_optimizer()
        self.network_summary()
        self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the " + self.hp["ae_type"] + " AE model...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()

            train_loss = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader, outdir)
            if self._early_stopping(estopping_limit):
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss.append(valid_losses[1].item())
            self.all_class_loss.append(valid_losses[2].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)

    def _early_stopping(self, early_stopping_limit) -> bool:
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.

        returns :: True for when the early stopping limit was exceeded
            and false otherwise.
        """
        if self.epochs_no_improve >= early_stopping_limit:
            return 1
        return 0
