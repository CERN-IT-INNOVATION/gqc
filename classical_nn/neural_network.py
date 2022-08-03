# Classical feedforward neural network model. To serve as a fair benchmark against
# the VQC and the Hybrid VQC.

from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

import os
import json
import matplotlib.pyplot as plt

from vqc_pennylane.terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need. Disabled to increase performance.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)


class NeuralNetwork(nn.Module):
    """
    TODO
    """
    def __init__(self, device="cpu", hpars={}):
        """
        TODO
        """
        super().__init__()
        self._hp = {
            "layers": [67, 64, 52, 44, 32, 24, 16, 1],
            "lr": 0.002,
            "batch_size": 128,
            "adam_betas": (0.9, 0.999),
            "out_activ": "nn.Sigmoid()",
        }
        self._device = device

        self._class_loss_function = nn.BCELoss(reduction="mean")
        self._hp.update((k, hpars[k]) for k in self._hp.keys() & hpars.keys())

        exec("self._out_activ = " + self._hp["out_activ"])

        self.best_valid_loss = 9999
        self.all_train_loss = []
        self.all_valid_loss = []
        self.epochs_no_improve = 0

        self._network = self.__construct_network()

    def __construct_network(self) -> nn.Sequential:
        """
        Construct the fully connected network.

        Returns: Pytorch sequence of layers of the network.
        """
        layers = []
        layer_nbs = range(len(self._hp["layers"]))
        for idx in layer_nbs:
            layers.append(nn.Linear(self._hp["layers"][idx], 
                                    self._hp["layers"][idx + 1]))
            if idx == len(self._hp["layers"]) - 2:
                layers.append(self._out_activ)
                break
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def instantiate_adam_optimizer(self):
        """Instantiate the optimizer object, used in the training of the model."""
        self = self.to(self._device)
        self.optimizer = optim.Adam(self.parameters(), lr=self._hp["lr"], betas=self._hp["adam_betas"])

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the network."""
        return self._network(x)

    def compute_loss(self, x_data: np.ndarray, y_data:np.ndarray) -> float:
        """
        Compute the loss of a forward pass.
        
        Args:
            x_data: Array of the input data.
            y_data: Array of the input data labels.

        Returns: The computed loss function value.
        """
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data).to(self._device)
        if isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data).to(self._device)

        output = self.forward(x_data.float())
        return self._class_loss_function(output.flatten(), y_data.float())

    @staticmethod
    def print_losses(epoch, epochs: int, train_loss: torch.Tensor, 
                     valid_loss: torch.Tensor):
        """
        Prints the training and validation losses in a nice format.
        
        Args:
            epoch: Current epoch.
            epochs: Total number of epochs.
            train_loss: The computed training loss pytorch object.
            valid_loss: The computed validation loss pytorch object.
        """
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train loss (average) = {train_loss.item():.8f}"
        )
        print(f"Epoch : {epoch + 1}/{epochs}, " f"Valid loss = {valid_loss.item():.8f}")

    @staticmethod
    def print_summary(torch_object):
        """
        Prints a neat summary the model with all the layers and activations.
        Can be the architecture of the network or the optimizer object.
        """
        try:
            summary(
                torch_object,
                show_input=True,
                show_hierarchical=False,
                print_summary=True,
                max_depth=1,
                show_parent_layers=False,
            )
        except Exception as e:
            print(e)
            print(tcols.WARNING + "Net summary failed!" + tcols.ENDC)

    def network_summary(self):
        """
        Prints a summary of the entire ae network.
        """
        print(tcols.OKGREEN + "Network summary:" + tcols.ENDC)
        self.print_summary(self._network)
        print("\n\n")

    def optimizer_summary(self):
        """
        Prints a summary of the optimizer that is used in the training.
        """
        print(tcols.OKGREEN + "Optimizer summary:" + tcols.ENDC)
        print(self.optimizer)
        print("\n\n")

    def save_best_loss_model(self, valid_loss: float, outdir: str):
        """
        Prints a message and saves the optimised model with the best loss.
        
        Args:
            valid_loss: Float of the validation loss.
            outdir: Directory where the best model is saved.
        """
        if self.best_valid_loss > valid_loss:
            self.epochs_no_improve = 0
            self.best_valid_loss = valid_loss

            print(tcols.OKGREEN + f"New min: {self.best_valid_loss:.2e}" + tcols.ENDC)
            if outdir is not None:
                torch.save(self.state_dict(), outdir + "best_model.pt")
        else:
            self.epochs_no_improve += 1

    def _early_stopping(self, early_stopping_limit) -> bool:
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.

        Returns: True for when the early stopping limit was exceeded and 
                 false otherwise.
        """
        if self.epochs_no_improve >= early_stopping_limit:
            return 1
        return 0

    @torch.no_grad()
    def valid(self, valid_loader: DataLoader, outdir: str) -> float:
        """
        Evaluate the validation loss for the model and save the model if a new
        new minimum is found.
        
        Args:
            valid_loader: Pytorch data loader with the validation data.
            outdir: Output folder where to save the model.

        Returns: Pytorch loss object of the validation loss.
        """
        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self._device)
        self.eval()

        loss = self.compute_loss(x_data_valid, y_data_valid)
        self.save_best_loss_model(loss, outdir)

        return loss

    def _train_batch(self, x_batch: torch.Tensor, 
                     y_batch: torch.Tensor = None) -> float:
        """
        Train the model on a batch and evaluate the loss.
        Propagate this backwards for minimum train_loss.
        
        Args:
            x_batch: Pytorch batch object with the data.
            y_batch: Pytorch batch object with the target.

        Returns: Training loss over a batch.
        """
        feature_size = x_batch.shape[1]
        init_feats = x_batch.view(-1, feature_size).to(self._device)
        if y_batch is not None:
            y_batch = y_batch.to(self._device)

        loss = self.compute_loss(init_feats, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_all_batches(self, train_loader: DataLoader) -> float:
        """
        Train the model on all the batches.
        
        Args:
            train_loader: Pytorch loader object with the training data.
        
        Returns: The training loss averaged over all the batches in an epoch.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        for batch in train_loader:
            x_batch, y_batch = batch
            batch_loss = self._train_batch(x_batch, y_batch)
            batch_loss_sum += batch_loss
            nb_of_batches += 1
        return batch_loss_sum / nb_of_batches

    def train_model(self, train_loader: DataLoader, valid_loader:DataLoader, 
                    epochs: int, early_stopping_limit: int, outdir: str):
        """
        Train the neural network.
        
        Args:
            train_loader: Pytorch data loader with the training data.
            valid_loader: Pytorch data loader with the validation data.
            epochs: The number of epochs to train for.
            outdir: The output dir where to save the training results.
        """
        self.instantiate_adam_optimizer()
        self.network_summary()
        self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the fully connected feed-forward neural network...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()
            train_loss = self._train_all_batches(train_loader)
            valid_loss = self.valid(valid_loader, outdir)
            if self._early_stopping(early_stopping_limit):
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_loss.item())
            self.print_losses(epoch, epochs, train_loss, valid_loss)

    def loss_plot(self, outdir: str):
        """
        Plots the loss for each epoch for the training and validation data.
        
        Args:
            outdir: Directory where to save the loss plot.
        """
        epochs = list(range(len(self.all_train_loss)))
        plt.plot(
            epochs,
            self.all_train_loss,
            color="gray",
            label="Training Loss (average)",
        )
        plt.plot(epochs, self.all_valid_loss, color="navy", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.text(
            np.min(epochs),
            np.max(self.all_train_loss),
            f"Min: {self.best_valid_loss:.2e}",
            verticalalignment="top",
            horizontalalignment="left",
            color="blue",
            fontsize=15,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

        plt.legend()
        plt.savefig(outdir + "loss_epochs.pdf")
        plt.close()

        print(tcols.OKGREEN + f"Loss vs epochs plot saved to {outdir}." + tcols.ENDC)

    def export_architecture(self, outdir: str):
        """
        Saves the structure of the NN to a file.
        
        Args:
            outdir: Directory where to save the architecture of the network.
        """
        with open(outdir + "model_architecture.txt", "w") as model_arch:
            print(self, file=model_arch)

    def export_hyperparameters(self, outdir: str):
        """
        Saves the hyperparameters of the model to a json file.
        
        Args:
            outdir: Directory where to save the json file.
        """
        file_path = os.path.join(outdir, "hyperparameters.json")
        params_file = open(file_path, "w")
        json.dump(self._hp, params_file)
        params_file.close()

    def load_model(self, model_path):
        """
        Loads the weights of a trained model saved in a .pt file.
        
        Args:
            model_path: Directory where a trained model was saved.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("∄ path.")
        self.load_state_dict(
            torch.load(model_path, map_location=torch.device(self._device))
        )

    @torch.no_grad()
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """
        Compute the model prediction for x_data.
        
        Args:
            x_data: Input data array for which the predicted label is computed.

        Returns: The predicted label of x_data.
        """
        x_data = torch.from_numpy(x_data).to(self._device)
        self.eval()
        return self.forward(x_data.float())
