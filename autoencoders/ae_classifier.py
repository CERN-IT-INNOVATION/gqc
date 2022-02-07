# Classifier autencoder. Different from the vanilla one since it has a
# classifier attached to the latent space, that does the classification
# for each batch latent space and outputs the binary cross-entropy loss
# that is then used to optimize the autoencoder as a whole.

import numpy as np
import torch
import torch.nn as nn

from .ae_vanilla import AE_vanilla
from .terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need. Disabled to increase performance.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)


class AE_classifier(AE_vanilla):
    def __init__(self, device="cpu", hparams={}):

        super().__init__(device, hparams)
        new_hp = {
            "ae_type": "classifier",
            "class_layers": [128, 64, 32, 16, 8, 1],
            "adam_betas": (0.9, 0.999),
            "class_weight": 0.5,
        }
        self.hp.update(new_hp)
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())

        self.class_loss_function = nn.BCELoss(reduction="mean")

        self.recon_loss_weight = 1 - self.hp["class_weight"]
        self.class_loss_weight = self.hp["class_weight"]
        self.all_recon_loss = []
        self.all_class_loss = []

        self.class_layers = [self.hp["ae_layers"][-1]] + self.hp["class_layers"]
        self.classifier = self.construct_classifier(self.class_layers)

    @staticmethod
    def construct_classifier(layers) -> nn.Sequential:
        """
        Construct the classifier neural network.
        @layers   :: Array of number of nodes for each layer.

        returns  :: Pytorch sequence of layers making the classifier NN.
        """
        dnn_layers = []

        for idx in range(len(layers)):
            dnn_layers.append(nn.Linear(layers[idx], layers[idx + 1]))
            if idx == len(layers) - 2:
                dnn_layers.append(nn.Sigmoid())
                break
            dnn_layers.append(nn.ReLU(True))

        return nn.Sequential(*dnn_layers)

    def forward(self, x):
        """
        Forward pass through the ae and the classifier.
        """
        latent = self.encoder(x)
        class_output = self.classifier(latent)
        reconstructed = self.decoder(latent)
        return latent, class_output, reconstructed

    def compute_loss(self, x_data, y_data) -> float:
        """
        Compute the loss of a forward pass through the ae and
        classifier. Combine the two losses and return the one loss.
        @x_data  :: Numpy array of the original input data.
        @y_data  :: Numpy array of the original target data.

        returns :: Float of the computed combined loss function value.
        """
        if type(x_data) is np.ndarray:
            x_data = torch.from_numpy(x_data).to(self.device)
        if type(y_data) is np.ndarray:
            y_data = torch.from_numpy(y_data).to(self.device)

        latent, classif, recon = self.forward(x_data.float())

        class_loss = self.class_loss_function(classif.flatten(), y_data.float())
        recon_loss = self.recon_loss_function(recon, x_data.float())

        return self.recon_loss_weight * recon_loss + self.class_loss_weight * class_loss

    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_losses):
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
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid loss = {valid_losses[0].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid recon loss (no weight) = {valid_losses[1].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid class loss (no weight) = {valid_losses[2].item():.8f}"
        )

    def network_summary(self):
        """
        Prints summary of entire AE + classifier network.
        """
        print(tcols.OKGREEN + "Encoder summary:" + tcols.ENDC)
        self.print_summary(self.encoder)
        print("\n")
        print(tcols.OKGREEN + "Classifier summary:" + tcols.ENDC)
        self.print_summary(self.classifier)
        print("\n")
        print(tcols.OKGREEN + "Decoder summary:" + tcols.ENDC)
        self.print_summary(self.decoder)
        print("\n\n")

    @torch.no_grad()
    def valid(self, valid_loader, outdir) -> list:
        """
        Evaluate the validation combined loss for the model and save the model
        if a new minimum in this combined and weighted loss is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        returns :: Pytorch loss object of the total validation loss,
            latent loss on the validation data, and recon loss on it.
        """
        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        y_data_valid = y_data_valid.to(self.device)
        self.eval()

        latent, classif, recon = self.forward(x_data_valid.float())

        recon_loss = self.recon_loss_function(x_data_valid.float(), recon)
        class_loss = self.class_loss_function(classif.flatten(), y_data_valid)

        valid_loss = (
            self.recon_loss_weight * recon_loss + self.class_loss_weight * class_loss
        )

        self.save_best_loss_model(valid_loss, outdir)

        return valid_loss, recon_loss, class_loss

    def train_all_batches(self, train_loader) -> float:
        """
        Train the autoencoder on all the batches.
        @train_loader :: Pytorch loader object with the training data.

        returns :: The normalised training loss over all the batches.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        for batch in train_loader:
            x_batch, y_batch = batch
            batch_loss = self.train_batch(x_batch, y_batch)
            batch_loss_sum += batch_loss
            nb_of_batches += 1

        return batch_loss_sum / nb_of_batches

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):
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
            if self.early_stopping():
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss.append(valid_losses[1].item())
            self.all_class_loss.append(valid_losses[2].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)

    @torch.no_grad()
    def predict(self, x_data) -> np.ndarray:
        """
        Compute the prediction of the autoencoder.
        @x_data :: Input array to pass through the autoencoder.

        returns :: Lists with the latent space of the ae, the
            reconstructed data, and the classifier output.
        """
        x_data = torch.from_numpy(x_data).to(self.device)
        self.eval()
        latent, classif, recon = self.forward(x_data.float())

        latent = latent.cpu().numpy()
        classif = classif.cpu().numpy()
        recon = recon.cpu().numpy()
        return latent, recon, classif
