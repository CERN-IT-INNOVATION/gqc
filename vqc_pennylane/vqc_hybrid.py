# Hybrid VQC.
import torch
from torch import nn
import pennylane as pnl
import numpy as np
import pennylane.numpy as pnp

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
    classifier. The latter can be constructed from the former by "chopping"
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
            "hybrid": True,
            "nqubits": 4,
            "nfeatures": 16,
            "fmap": "zzfm",
            "vform": "two_local",
            "vform_repeats": 4,
        }
        self.hp.update(new_hp)
        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())

        self._qdevice = qdevice
        self._layers = self._check_compatibility(
            self.hp["nqubits"], self.hp["nfeatures"]
        )
        self._diff_method = self._select_diff_method(hpars)
        self.epochs_no_improve = 0

        self._vqc_nweights = vf.vforms_weights(
            self.hp["vform"], self.hp["vform_repeats"], self.hp["nqubits"]
        )
        self._weight_shape = {"weights": (self._layers, self._vqc_nweights)}

        self._circuit = pnl.qnode(self._qdevice, interface="torch",
                                  diff_method = self._diff_method)(
            self.__construct_classifier
        )
        self.classifier = pnl.qnn.TorchLayer(self._circuit, self._weight_shape)
        del self.class_loss_function
        self.class_loss_function = self._shifted_bce

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
        
        return pnl.expval(pnl.PauliZ(0))
    
    def _shifted_bce(self, x, y):
        """
        Shift the input given to this method and calculate the binary cross entropy
        loss. This shift is required to have the output of the VQC model in [0,1].
        Args:
            x (torch.tensor): Data point/batch to evaluate the loss on.
            y (torch.tensor): Corresponding labels of the point/batch.
        Returns:
            The binary cross entropy loss computed on the given data.
        """
        return nn.BCELoss(reduction="mean")((x+1)/2, y)

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
    
    @staticmethod
    def _select_diff_method(hpars: dict) -> str:
        """Checks if a differentiation method for the quantum circuit is specified
        by the user. If not, 'best' is selected as the differentiation method.

        Args:
            args: Arguments given to the vqc by the user, specifiying various hps.

        Returns:
            String that specifies which differentiation method to use.
        """
        if "diff_method" in hpars:
            return hpars["diff_method"]

        return "best"

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

    def __initialise_weights(self) -> np.ndarray:
        """Method to initialise random weights only for drawing the circuit."""
        weights = 0.01 * np.random.randn(self._layers, self._vqc_nweights)
        return weights

    def draw(self):
        """
        Draws the circuit using dummy parameters.
        Parameterless implementation is not yet available in pennylane,
        and it seems not feasible either by the way pennylane is constructed.
        """
        drawing = pnl.draw(self._circuit)
        print(tcols.OKGREEN)
        print(drawing([0] * int(self.hp["nfeatures"]), self.__initialise_weights()))
        print(tcols.ENDC)
