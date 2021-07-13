# The vanilla autoencoder architecture. Reduces the number
# of features from 67 down to 16 by using a combination of linear and ELU
# layers. The loss function of the autoencoder is the MSE between the
# histograms reconstructed by the decoder and the original variables.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE(nn.Module):
    def __init__(self, device, layers, lr, en_activ=None, dec_activ=None):

        super(AE, self).__init__()
        self.lr                  = lr
        self.layers              = layers
        self.device              = device
        self.recon_loss_function = nn.MSELoss(reduction='mean')

        self.best_valid_loss   = 9999
        self.all_train_loss    = []
        self.all_valid_loss    = []

        encoder_layers = self.construct_encoder(layers, en_activ)
        self.encoder   = nn.Sequential(*encoder_layers)

        decoder_layers = self.construct_decoder(layers, dec_activ)
        self.decoder   = nn.Sequential(*decoder_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def construct_encoder(layers, en_activ):
        """
        Construct the encoder layers.
        """
        enc_layers = []
        layer_nbs = range(len(layers))
        for idx in layer_nbs:
            enc_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            if idx == len(layers) - 2 and en_activ is None: break
            if idx == len(layers) - 2: enc_layers.append(en_activ); break
            enc_layers.append(nn.ELU(True))

        return enc_layers

    @staticmethod
    def construct_decoder(layers, dec_activ):
        """
        Construct the decoder layers.
        """
        dec_layers = []
        layer_nbs = reversed(range(len(layers)))
        for idx in layer_nbs:
            dec_layers.append(nn.Linear(layers[idx], layers[idx-1]))
            if idx == 1 and dec_activ is None: break
            if idx == 1 and dec_activ: dec_layers.append(dec_activ); break
            dec_layers.append(nn.ELU(True))

        return dec_layers

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        # Evaluate the validation loss for the model and save if new minimum.

        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        self.eval()

        latent, recon = self.forward(x_data_valid.float())

        recon_loss = self.recon_loss_function(recon, x_data_valid.float())

        if recon_loss < self.best_valid_loss: self.best_valid_loss = recon_loss

        if outdir is not None and self.best_valid_loss == recon_loss:
            print(f'\033[92mNew min loss: {self.best_valid_loss:.2e}\033[0m')
            torch.save(self.state_dict(), outdir + 'best_model.pt')

        self.all_valid_loss.append(recon_loss)

    def print_losses(self, epoch, epochs):
        print(f"Epoch : {epoch + 1}/{epochs}, "
                  f"Train loss (last batch) = {self.all_train_loss[epoch]:.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
                  f"Valid loss = {self.all_valid_loss[epoch]:.8f}")

    def train_batch(self, x_batch):
        # Train the model on a batch and evaluate the different kinds of losses.
        # Propagate this backwards for minimum train_loss.

        feature_size  = x_batch.shape[1]
        init_feats    = x_batch.view(-1, feature_size).to(self.device)
        latent, recon = self.forward(init_feats.float())

        recon_loss = self.recon_loss_function(recon, init_feats.float())

        self.optimizer.zero_grad()
        recon_loss.backward()
        self.optimizer.step()

        return recon_loss

    def train_all_batches(self, train_loader):

        for batch in train_loader:
            x_batch, y_batch      = batch
            last_batch_train_loss = self.train_batch(x_batch)

        self.all_train_loss.append(last_batch_train_loss.item())

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):

        print('\033[96mTraining the classifier AE model...\033[0m')

        for epoch in range(epochs):
            self.train()

            self.train_all_batches(train_loader)
            self.valid(valid_loader, outdir)

            self.print_losses(epoch, epochs)

        return self.all_train_loss, self.all_valid_loss, self.best_valid_loss
