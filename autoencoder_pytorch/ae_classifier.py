# Classifier autencoder. Different from the vanilla one since it has a
# classifier attached to the latent space, that does the classification
# for each batch latent space and outputs the binary cross-entropy loss
# that is then used to optimize the autoencoder as a whole.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ae_vanilla import AE_vanilla
from terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_classifier(AE_vanilla):
    def __init__(self, device, layers, lr, en_activ, dec_activ, class_layers,
        loss_weight, **kwargs):

        super().__init__(device, layers, lr, en_activ, dec_activ)

        self.class_layers        = class_layers
        self.class_loss_function = nn.BCELoss(reduction='mean')
        self.loss_weight         = loss_weight

        (self.class_layers).insert(0, layers[-1])
        self.class_layers = self.construct_classifier(self.class_layers)
        self.classifier   = nn.Sequential(*class_layers)
        self = self.to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def construct_classifier(layers):
        # Construct the classifier layers.
        dnn_layers = []
        # dnn_layers.append(nn.BatchNorm1d(layers[0]))

        for idx in range(len(layers)):
            dnn_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            if idx == len(layers) - 2: dnn_layers.append(nn.Sigmoid()); break

            # dnn_layers.append(nn.BatchNorm1d(layers[idx+1]))
            # dnn_layers.append(nn.Dropout(0.5))
            dnn_layers.append(nn.LeakyReLU(0.2))

        return dnn_layers

    def forward(self, x):
        latent        = self.encoder(x)
        class_output  = self.classifier(latent)
        reconstructed = self.decoder(latent)
        return latent, class_output, reconstructed

    def compute_loss(self, x_data, y_data):
        if type(x_data) is np.ndarray:
            x_data = torch.from_numpy(x_data).to(self.device)
        if type(y_data) is np.ndarray:
            y_data = torch.from_numpy(y_data).to(self.device)

        latent, classif, recon = self.forward(x_data.float())

        recon_loss = self.recon_loss_function(recon, x_data.float())
        class_loss = self.class_loss_function(classif.flatten(), y_data.float())

        return (1 - self.loss_weight)*recon_loss + self.loss_weight*class_loss

    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_losses):

        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Train loss (last batch) = {train_loss.item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid loss = {valid_losses[0].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid recon loss = {valid_losses[1].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid class loss = {valid_losses[2].item():.8f}")

    def network_summary(self):
        print(tcols.OKGREEN + "Encoder summary:" + tcols.ENDC)
        self.print_summary(self.encoder, self.device)
        print('\n')
        print(tcols.OKGREEN + "Classifier summary:" + tcols.ENDC)
        self.print_summary(self.classifier, self.device)
        print('\n')
        print(tcols.OKGREEN + "Decoder summary:" + tcols.ENDC)
        self.print_summary(self.decoder, self.device)
        print('\n\n')

    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        # Evaluate the validation loss for the model and save if new minimum.

        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        y_data_valid = y_data_valid.to(self.device)
        self.eval()

        latent, classif, recon = self.forward(x_data_valid.float())

        recon_loss = self.recon_loss_function(x_data_valid.float(), recon)
        class_loss = self.class_loss_function(classif.flatten(), y_data_valid)
        valid_loss = self.compute_loss(x_data_valid, y_data_valid)

        self.save_best_loss_model(valid_loss, outdir)

        return valid_loss, recon_loss, class_loss

    def train_all_batches(self, train_loader):

        batch_loss_sum = 0
        nb_of_batches  = 0
        for batch in train_loader:
            x_batch, y_batch = batch
            batch_loss       = self.train_batch(x_batch, y_batch)
            batch_loss_sum   += batch_loss
            nb_of_batches    += 1

        return batch_loss_sum/nb_of_batches

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):

        self.network_summary(); self.optimizer_summary()
        print(tcols.OKCYAN + "Training the classifier AE model..." + tcols.ENDC)
        all_train_loss = []; all_valid_loss = []

        for epoch in range(epochs):
            self.train()

            train_loss   = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader,outdir)

            if early_stopping():
                return all_train_loss, all_valid_loss, self.best_valid_loss

            all_train_loss.append(train_loss.item())
            all_valid_loss.append(valid_losses[0].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)

        return all_train_loss, all_valid_loss, self.best_valid_loss

    @torch.no_grad()
    def predict(self, x_data):
        # Compute the prediction of the autoencoder, given input np array x.
        x_data = torch.from_numpy(x_data).to(self.device)
        self.eval()
        latent, classif, recon = self.forward(x_data.float())

        latent  = latent.cpu().numpy()
        classif = classif.cpu().numpy()
        recon   = recon.cpu().numpy()
        return latent, recon, classif
