# SVM autencoder. Different from the vanilla one since it has a
# SVM attached to the latent space, that does the classification
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

# WORK IN PROGRESS.
class AE_svm(AE_vanilla):
    def __init__(self, device, layers, lr, en_activ, dec_activ, loss_weight):

        super().__init__(device, layers, lr, en_activ, dec_activ)

        self.class_loss_function = self.hinge_loss
        self.loss_weight         = loss_weight

        self.svm = nn.Sequential(nn.Linear(16, 1))
        self     = self.to(device)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        latent        = self.encoder(x)
        classif       = self.svm(latent)
        reconstructed = self.decoder(latent)
        return latent, classif, reconstructed

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
    def hinge_loss(output, y):
        return torch.mean(torch.clamp(1 - output * y, min=0))

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
        self.print_summary(self.svm, self.device)
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

        if valid_loss < self.best_valid_loss: self.best_valid_loss = valid_loss

        if outdir is not None and self.best_valid_loss == valid_loss:
            print(tcols.OKGREEN + f"New min loss: {self.best_valid_loss:.2e}" +
                  tcols.ENDC)
            torch.save(self.state_dict(), outdir + 'best_model.pt')

        return valid_loss, recon_loss, class_loss

    def train_batch(self, x_batch, y_batch):
        # Train the model on a batch and evaluate the different kinds of losses.
        # Propagate this backwards for minimum train_loss.

        feature_size = x_batch.shape[1]
        init_feats   = x_batch.view(-1, feature_size).to(self.device)
        y_batch      = y_batch.to(self.device)

        batch_train_loss = self.compute_loss(init_feats, y_batch)

        self.optimizer.zero_grad()
        batch_train_loss.backward()
        self.optimizer.step()

        return batch_train_loss

    def train_all_batches(self, train_loader):

        for batch in train_loader:
            x_batch, y_batch      = batch
            last_batch_train_loss = self.train_batch(x_batch, y_batch)

        return last_batch_train_loss

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):

        self.network_summary()
        self.optimizer_summary()
        print(tcols.OKCYAN + "Training the SVM AE model..." + tcols.ENDC)
        all_train_loss = []
        all_valid_loss = []

        for epoch in range(epochs):
            self.train()

            train_loss   = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader,outdir)

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
