import pennylane as pnl
import numpy as np
import torch.nn as nn
from pennylane.optimize import AdamOptimizer

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
                 vform_repeats=4, optimizer=None, lr=0.001):
        """
        @nqubits   :: Number of qubits the circuit should be made of.
        @nfeatures :: Number of features in the training data set.
        @fmap      :: String name of the feature map to use.
        @vform     :: String name of the variational form to use.
        """
        self._layers = self._check_compatibility(nqubits, nfeatures)
        self._nfeatures = nfeatures
        self._nqubits = nqubits
        self._vform_repeats = vform_repeats
        self._nweights = vf.vforms_weights(vform, vform_repeats, nqubits)

        np.random.seed(123)
        self._weights = 0.01*np.random.randn(self._nweights, self._layers,
                                             requires_grad=True)
        self._optimiser = self._choose_optimiser(optimiser, lr)
        self._class_loss_function = nn.BCELoss(reduction="mean")
        self._epochs_no_improve = 0

        self._device = pnl.device("default.qubit", wires=nqubits)
        self._circuit = pnl.qnode(self._device)(self._qcircuit)

    def _qcircuit(self, inputs, weights):
        """
        The quantum circuit builder.
        @inputs  :: The inputs taken by the feature maps.
        @weights :: The weights of the variational forms used.

        returns :: Measurement of the first qubit of the quantum circuit.
        """
        for layer_nb in range(self._layers):
            start_feature = layer_nb*self._nqubits
            end_feature = self._nqubits*(layer_nb + 1)

            fm.zzfm(self._nqubits, inputs[start_feature:end_feature])
            vf.two_local(self._nqubits, weights[layer_nb],
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
        print(drawing([0]*int(self._nfeatures), self._weights))
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

    @staticmethod
    def _choose_optimiser(choice, lr):
        """
        Choose an optimiser to use in the training of the vqc.
        @choice :: String of the optimiser name you want to use to train vqc.
        @lr     :: Learning rate for the optimiser.
        """
        if choice is None: return None

        switcher = {
            "adam" : lambda : AdamOptimizer(stepsize=lr)
        }
        optimiser = switcher.get(choice, lambda: None)()
        if optimiser is None:
            raise TypeError("Specified optimiser is not an option atm!")

        return optimiser

    def _early_stopping(self, early_stopping_limit) -> bool:
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.

        returns :: True for when the early stopping limit was exceeded
            and false otherwise.
        """
        if self._epochs_no_improve >= early_stopping_limit:
            return 1
        return 0

    def forward(self, x_data):
        return [self._qcircuit(x, self._weights) for x in x_data]

    def _save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimised model with the best loss.
        @valid_loss :: Float of the validation loss.
        @outdir     :: Directory where the best model is saved.
        """
        if self.best_valid_loss > valid_loss:
            self._epochs_no_improve = 0
            print(tcols.OKGREEN + f"New min: {self.best_valid_loss:.2e}" +
                  tcols.ENDC)

            self._best_valid_loss = valid_loss
            if outdir is not None:
                torch.save(self.state_dict(), outdir + "best_model.pt")

        else:
            self.epochs_no_improve += 1

    @staticmethod
    def _objective_function(weights, x_batch, y_batch):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        self._weights = weights
        predictions = self.forward(x_batch)

        return self._class_loss_function(predictions, y_batch)

    def _validate(self, valid_loader, outdir):
        """
        Calculate the loss on a validation data set.
        """
        x_valid, y_valid = valid_loader

        loss = self._objective_function(self._weights, x_valid, y_valid)
        self._save_best_loss_model(loss, outdir)

        return loss

    def _train_batch(self, x_batch, y_batch):
        """
        Train on one batch.
        """
        weights, _, _ = self._optimiser.step(self._objective_function,
                                             self._weights, x_batch, y_batch)
        self._weights = weights
        loss = self._objective_function(self._weights, x_batch, y_batch)

        return loss

    def _train_all_batches(self, train_loader):
        """
        Train on the full data set.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        x_train, y_train = train_loader
        for x_batch, y_batch in zip(x_train, y_train):
            batch_loss = self._train_batch(x_batch, y_batch)
            batch_loss_sum += batch_loss
            nb_of_batches += 1

        return batch_loss_sum / nb_of_batches


    def train_vqc(self, train_loader, valid_loader, epochs, estopping_limit,
                  outdir):
        """
        Train an instantiated vqc algorithm.
        """
        print(tcols.OKCYAN + "Training the vqc..." + tcols.ENDC)

        for epoch in range(epochs):
            train_loss = self._train_all_batches(train_loader)
            valid_loss = self._validate(valid_loader, outdir)
            if self.early_stopping(estopping_limit):
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_loss.item())
            self.print_losses(epoch, epochs, train_loss, valid_loss)
