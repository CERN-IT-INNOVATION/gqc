# Classifier autencoder. Different from the vanilla one since it has a
# classifier attached to the latent space, that does the classification
# for each batch latent space and outputs the binary cross-entropy loss
# that is then used to optimize the autoencoder as a whole.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from ae_vanilla import AE_vanilla
from terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_classifier(AE_vanilla):
    def __init__(self, device = 'cpu', hparams = {}):

        super().__init__(device, hparams)
        new_hp = {
            "ae_type"      : "classifier",
            "class_layers" : [128, 64, 32, 16, 8, 1],
            "adam_betas"   : (0.9, 0.999),
            "loss_weight"  : 0.5
        }
        self.hp.update(new_hp)
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())

        self.class_loss_function = nn.BCELoss(reduction='mean')

        self.recon_loss_weight = 1 - self.hp['loss_weight']
        self.class_loss_weight = self.hp['loss_weight']
        self.all_recon_loss = []
        self.all_class_loss = []

        self.class_layers = [self.hp['ae_layers'][-1]] + self.hp['class_layers']
        self.classifier   = self.construct_classifier(self.class_layers)

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

        return nn.Sequential(*dnn_layers)

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

        class_loss = self.class_loss_function(classif.flatten(), y_data.float())
        recon_loss = self.recon_loss_function(recon, x_data.float())

        return self.recon_loss_weight*recon_loss + \
               self.class_loss_weight*class_loss

    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_losses):

        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Train loss (average) = {train_loss.item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid loss = {valid_losses[0].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid recon loss (no weight) = {valid_losses[1].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid class loss (no weight) = {valid_losses[2].item():.8f}")

    def network_summary(self):
        print(tcols.OKGREEN + "Encoder summary:" + tcols.ENDC)
        self.print_summary(self.encoder)
        print('\n')
        print(tcols.OKGREEN + "Classifier summary:" + tcols.ENDC)
        self.print_summary(self.classifier)
        print('\n')
        print(tcols.OKGREEN + "Decoder summary:" + tcols.ENDC)
        self.print_summary(self.decoder)
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

        valid_loss = self.recon_loss_weight * recon_loss + \
                     self.class_loss_weight * class_loss

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

        self.instantiate_adam_optimizer()
        self.network_summary(); self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the " + self.hp['ae_type'] + " AE model...")
        print(tcols.ENDC)
        all_train_loss = []; all_valid_loss = []

        for epoch in range(epochs):
            self.train()

            train_loss   = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader,outdir)
            if self.early_stopping(): break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss.append(valid_losses[1].item())
            self.all_class_loss.append(valid_losses[2].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)

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
