# VQC implemented in pennylane.
import os
import json
import pennylane as pnl

from pennylane import numpy as np
import autograd.numpy as anp
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt

from . import feature_maps as fm
from . import variational_forms as vf
from .terminal_colors import tcols


class VQC:
    def __init__(self, qdevice: pnl.device, hpars: dict):
        """
        Variational quantum circuit, implemented using the pennylane python
        package. This is a trainable quantum circuit. It is composed of a feature
        map and a variational form, which are implemented in their eponymous
        files in the same directory.

        Args:
            qdevice: String containing what kind of device to run the
                      quantum circuit on: simulation, or actual computer?
            hpars: Dictionary of the hyperparameters to configure the vqc.
        """
        self._hp = {
            "hybrid": False,
            "nqubits": 4,
            "nfeatures": 16,
            "fmap": "zzfm",
            "vform": "two_local",
            "vform_repeats": 4,
            "optimiser": "adam",
            "lr": 0.001,
        }

        self._hp.update((k, hpars[k]) for k in self._hp.keys() & hpars.keys())
        self._qdevice = qdevice
        self._layers = self._check_compat(self._hp["nqubits"], self._hp["nfeatures"])
        self._nweights = vf.vforms_weights(
            self._hp["vform"], self._hp["vform_repeats"], self._hp["nqubits"]
        )

        np.random.seed(123)
        self._weights = 0.01 * np.random.randn(
            self._layers, self._nweights, requires_grad=True
        )

        self._diff_method = self.select_diff_method(hpars)
        self._optimiser = self._choose_optimiser(self._hp["optimiser"], self._hp["lr"])
        self._class_loss_function = self._binary_cross_entropy
        self._epochs_no_improve = 0
        self._best_valid_loss = 999
        self.all_train_loss = []
        self.all_valid_loss = []

        self._circuit = pnl.qnode(self._qdevice, diff_method=self._diff_method)(
            self._qcircuit
        )

    def _qcircuit(self, inputs, weights):
        """
        The quantum circuit builder.
        @inputs  :: The inputs taken by the feature maps.
        @weights :: The weights of the variational forms used.

        returns :: Measurement of the first qubit of the quantum circuit.
        """
        for layer_nb in range(self._layers):
            start_feature = layer_nb * self._hp["nqubits"]
            end_feature = self._hp["nqubits"] * (layer_nb + 1)
            fm.zzfm(self._hp["nqubits"], inputs[start_feature:end_feature])
            vf.two_local(
                self._hp["nqubits"],
                weights[layer_nb],
                repeats=self._hp["vform_repeats"],
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
    def subforms(self):
        return self._subforms

    @property
    def nweights(self):
        return self._nweights

    @property
    def best_valid_loss(self):
        return self._best_valid_loss

    @property
    def total_weights(self):
        return self._nweights*self._layers

    def draw(self):
        """
        Draws the circuit using dummy parameters.
        Parameterless implementation is not yet available in pennylane,
        and it seems not feasible either by the way pennylane is constructed.
        """
        drawing = pnl.draw(self._circuit)
        print(tcols.OKGREEN)
        print(drawing([0] * int(self._hp["nfeatures"]), self._weights))
        print(tcols.ENDC)

    @staticmethod
    def _check_compat(nqubits, nfeatures):
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
    def select_diff_method(hpars: dict) -> str:
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

    @staticmethod
    def _choose_optimiser(choice, lr):
        """
        Choose an optimiser to use in the training of the vqc.
        @choice :: String of the optimiser name you want to use to train vqc.
        @lr     :: Learning rate for the optimiser.
        """
        if choice is None:
            return None

        switcher = {"adam": lambda: AdamOptimizer(stepsize=lr)}
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
        return [self._circuit(x, self._weights) for x in x_data]

    @staticmethod
    def _binary_cross_entropy(y_preds, y_batch):
        """
        Binary cross entropy loss calculation.
        """
        eps = anp.finfo(np.float32).eps
        y_preds = anp.clip(y_preds, eps, 1 - eps)
        y_batch = anp.array(y_batch)
        bce_one = anp.array(
            [y * anp.log(pred + eps) for pred, y in zip(y_preds, y_batch)]
        )
        bce_two = anp.array(
            [(1 - y) * anp.log(1 - pred + eps) for pred, y in zip(y_preds, y_batch)]
        )

        bce = anp.array(bce_one + bce_two)

        return -anp.mean(bce)

    def compute_loss(self, x_batch, y_batch, weights=None):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        if not weights is None:
            self._weights = weights
        predictions = self.forward(x_batch)
        return self._class_loss_function(predictions, y_batch)

    def _validate(self, valid_loader, outdir):
        """
        Calculate the loss on a validation data set.
        """
        x_valid, y_valid = valid_loader
        loss = self.compute_loss(x_valid, y_valid, self._weights)
        self._save_best_loss_model(loss, outdir)

        return loss

    def _train_batch(self, x_batch, y_batch):
        """
        Train on one batch.
        """
        x_batch = np.array(x_batch[:, :], requires_grad=False)
        y_batch = np.array(y_batch[:], requires_grad=False)
        _, _, weights = self._optimiser.step(
            self.compute_loss, x_batch, y_batch, self._weights
        )
        self._weights = weights
        loss = self.compute_loss(x_batch, y_batch, self._weights)
        return loss

    def _train_all_batches(self, train_loader, batch_seed):
        """
        Train on the full data set. Add randomness.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        x_train, y_train = train_loader
        for x_batch, y_batch in zip(x_train, y_train):
            np.random.seed(batch_seed)
            perm = np.random.permutation(len(y_batch))
            x_batch = x_batch[perm]
            y_batch = y_batch[perm]
            batch_loss = self._train_batch(x_batch, y_batch)
            batch_loss_sum += batch_loss
            nb_of_batches += 1

        return batch_loss_sum / nb_of_batches

    def _print_total_weights(self):
        """Prints the total weights of the vqc model, to mimic the behaviour that is
        available out of the box for the pytorch counterpart of this hybrid vqc.
        """
        print("----------------------------------------")
        print(tcols.OKGREEN + "Total number of weights: " + tcols.ENDC +
              f"{self.total_weights}")
        print("----------------------------------------")

    def train_model(self, train_loader, valid_loader, epochs, estopping_limit, outdir):
        """Train an instantiated vqc algorithm.
        """
        self._print_total_weights()
        print(tcols.OKCYAN + "Training the vqc..." + tcols.ENDC)
        rng = np.random.default_rng(12345)
        batch_seeds = rng.integers(low=0, high=100,
                                   size=(train_loader[1].shape[0], epochs))

        for epoch in range(epochs):
            train_loss = self._train_all_batches(train_loader, batch_seeds[epoch])
            valid_loss = self._validate(valid_loader, outdir)
            if self._early_stopping(estopping_limit):
                break

            self.all_train_loss.append(train_loss)
            self.all_valid_loss.append(valid_loss)
            self._print_losses(epoch, epochs, train_loss, valid_loss)

    @staticmethod
    def _print_losses(epoch, epochs, train_loss, valid_loss):
        """
        Prints the training and validation losses in a nice format.
        @epoch      :: Int of the current epoch.
        @epochs     :: Int of the total number of epochs.
        @train_loss :: The computed training loss pytorch object.
        @valid_loss :: The computed validation loss pytorch object.
        """
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train loss (average) = {train_loss.item():.8f}"
        )
        print(f"Epoch : {epoch + 1}/{epochs}, " f"Valid loss = {valid_loss.item():.8f}")

    def loss_plot(self, outdir):
        """
        Plots the loss for each epoch for the training and validation data.
        @outdir :: Directory where to save the loss plot.
        """
        epochs = list(range(len(self.all_train_loss)))
        plt.plot(epochs, self.all_train_loss, color="gray", label="Train Loss (avg)")
        plt.plot(epochs, self.all_valid_loss, color="navy", label="Valid Loss")
        self._loss_plot_header(epochs, "blue", "white")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.savefig(outdir + "loss_epochs.pdf")
        plt.close()

        print(tcols.OKGREEN + f"Loss plot was saved to {outdir}!" + tcols.ENDC)

    def _loss_plot_header(self, epochs, writing_color, box_color):
        """The header of the loss plot, displaying the best obtained loss during the
        training of the vqc and the legend.

        Args:
            epochs (int): Number of epochs the vqc was training for.
            writing_color (str): Color of the writing in the header of the figure.
            box_color (str): Color of the box in which this header is put.
        """
        plt.text(
            np.min(epochs),
            np.max(self.all_train_loss),
            f"Min: {self._best_valid_loss:.2e}",
            verticalalignment="top",
            horizontalalignment="left",
            color=writing_color,
            fontsize=15,
            bbox={"facecolor": box_color, "alpha": 0.8, "pad": 5},
        )
        plt.legend()

    def _save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimised model with the best loss.
        @valid_loss :: Float of the validation loss.
        @outdir     :: Directory where the best model is saved.
        """
        if self._best_valid_loss > valid_loss:
            self._epochs_no_improve = 0
            self._best_valid_loss = valid_loss

            print(tcols.OKGREEN + f"New min: {self.best_valid_loss:.2e}" + tcols.ENDC)
            if outdir is not None:
                np.save(outdir + "best_model.npy", self._weights)
        else:
            self._epochs_no_improve += 1

    def export_hyperparameters(self, outdir):
        """
        Saves the hyperparameters of the model to a json file.
        @outdir :: Directory where to save the json file.
        """
        file_path = os.path.join(outdir, "hyperparameters.json")
        params_file = open(file_path, "w")
        json.dump(self._hp, params_file)
        params_file.close()

    def load_model(self, model_path):
        """
        Loads the weights of a trained model saved in a numpy file.
        @model_path :: Directory where a trained model was saved.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("∄ path.")
        self._weights = np.load(model_path)

    def predict(self, x_data) -> np.ndarray:
        """
        Compute the prediction of the vqc on a data array.
        @x_data :: Input array to pass through the vqc.

        returns :: The latent space of the ae and the reco data.
        """
        x_data = np.array(x_data[:, :], requires_grad=False)
        classification_output = self.forward(x_data)

        return [classification_output]
