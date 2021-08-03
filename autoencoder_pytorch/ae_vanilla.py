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

    def compute_loss(self, x_data, y_data=None):
        if type(x_data) is np.ndarray:
            x_data = torch.from_numpy(x_data).to(self.device)
        latent, recon = self.forward(x_data.float())
        return self.recon_loss_function(recon, x_data.float())

    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_loss):
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Train loss (last batch) = {train_loss:.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid loss = {valid_loss:.8f}")

    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        # Evaluate the validation loss for the model and save if new minimum.

        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        self.eval()

        loss = self.compute_loss(x_data_valid)

        if loss < self.best_valid_loss: self.best_valid_loss = loss

        if outdir is not None and self.best_valid_loss == loss:
            print(f'\033[92mNew min loss: {self.best_valid_loss:.2e}\033[0m')
            torch.save(self.state_dict(), outdir + 'best_model.pt')

        return loss

    def train_batch(self, x_batch):
        # Train the model on a batch and evaluate the different kinds of losses.
        # Propagate this backwards for minimum train_loss.

        feature_size  = x_batch.shape[1]
        init_feats    = x_batch.view(-1, feature_size).to(self.device)

        loss = self.compute_loss(init_feats)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_all_batches(self, train_loader):
        # Train the autoencoder on all the batches.
        for batch in train_loader:
            x_batch, y_batch      = batch
            last_batch_train_loss = self.train_batch(x_batch)

        return last_batch_train_loss

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):

        print('\033[96mTraining the vanilla AE model...\033[0m')
        all_train_loss = []
        all_valid_loss = []

        for epoch in range(epochs):
            self.train()

            train_loss = self.train_all_batches(train_loader)
            valid_loss = self.valid(valid_loader, outdir)

            all_train_loss.append(train_loss.item())
            all_valid_loss.append(valid_loss.item())
            self.print_losses(epoch,epochs,train_loss.item(),valid_loss.item())

        return all_train_loss, all_valid_loss, self.best_valid_loss

    @torch.no_grad()
    def predict(self, x_data):
        # Compute the prediction of the autoencoder, given input np array x.
        x_data = torch.from_numpy(x_data).to(self.device)
        self.eval()
        latent, reconstructed = self.forward(x_data.float())

        latent        = latent.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        return latent, reconstructed
