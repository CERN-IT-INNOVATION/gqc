# Standard VAE architecture inherited from the vanilla architecture.
# The latent space is standardized to be gaussian, for easier sampling.
# In principle, it can be standardized to be any kind of distribution,
# but Gaussian is the usual one found in literature.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as dist

from ae_vanilla import AE_vanilla
from terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_variational(AE_vanilla):
    def __init__(self, device='cpu', hparams={}):

        super().__init__(device, hparams)
        new_hp = {
            "ae_type"     : "variational",
            "adam_betas"  : (0.9, 0.999),
            "loss_weight" : 0.5
        }

        self.hp.update(new_hp)
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())

        self.laten_loss_function = nn.KLDivLoss(reduction='batchmean')
        self.desired_latent_dist = dist.Normal(0, 1)

        self.recon_loss_weight = 1 - self.hp['loss_weight']
        self.laten_loss_weight = self.hp['loss_weight']

        del self.encoder

        self.encoder = self.construct_encoder(self.hp['ae_layers'])
        self.encode_mean = \
            nn.Linear(self.hp['ae_layers'][-2], self.hp['ae_layers'][-1])
        self.encode_logvar = \
            nn.Linear(self.hp['ae_layers'][-2], self.hp['ae_layers'][-1])

        self.all_recon_loss = []
        self.all_laten_loss = []

    @staticmethod
    def construct_encoder(layers, en_activ=None):
        """
        Construct the variational encoder.
        @layers    :: Array of number of nodes for each layer.
        @enc_activ :: Pytorch object for encoder activation function.

        @returns  :: Pytorch sequence of layers making the encoder.
        """
        enc_layers = []
        layer_nbs = range(len(layers))
        for idx in layer_nbs:
            if idx == len(layers) - 2 and en_activ is None: break
            if idx == len(layers) - 2: enc_layers.append(en_activ); break
            enc_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            enc_layers.append(nn.ELU(True))

        return nn.Sequential(*enc_layers)

    def reparametrize(self, mu, log_var):
        """
        Implement the reparametrization trick to be able to sample from the
        produced encoder distributions (latent space).
        @mu      :: The mean of the encoder distribution.
        @log_var :: The log of the variance of the encoder distribution.
        """
        std = torch.exp(0.5*log_var)
        eps = self.desired_latent_dist.sample(std.shape).to(self.device)
        latent_space_sample = mu + eps*log_var

        return latent_space_sample

    def forward(self, x):
        """
        Forward pass through the variational ae.
        """
        pre_latent_x = self.encoder(x)
        mu      = self.encode_mean(pre_latent_x)
        log_var = self.encode_logvar(pre_latent_x)

        latent        = self.reparametrize(mu, log_var)
        reconstructed = self.decoder(latent)

        return latent, reconstructed

    def compute_loss(self, x_data, y_data):
        """
        Compute the loss of a forward pass through the variational ae.
        Combine the reconstruction loss with the KL divergence loss to get
        the total loss which is then propagated backwards..
        @x_data  :: Numpy array of the original input data.
        @y_data  :: Numpy array of the original target data.

        @returns :: Float of the computed combined loss function value.
        """
        if type(x_data) is np.ndarray:
            x_data = torch.from_numpy(x_data).to(self.device)
        latent, recon = self.forward(x_data.float())

        samples = self.desired_latent_dist.sample(latent.shape)
        comparison_probs = func.softmax(samples, dim=1).to(self.device)
        latent_log_probs = func.log_softmax(latent, dim=1)

        laten_loss = self.laten_loss_function(latent_log_probs,comparison_probs)
        recon_loss = self.recon_loss_function(recon, x_data.float())

        return self.recon_loss_weight*recon_loss + \
               self.laten_loss_weight*laten_loss

    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_losses):
        """
        Prints the training and validation losses in a nice format.
        @epoch      :: Int of the current epoch.
        @epochs     :: Int of the total number of epochs.
        @train_loss :: The computed training loss pytorch object.
        @valid_loss :: The computed validation loss pytorch object.
        """
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Train loss (average) = {train_loss.item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid loss = {valid_losses[0].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid recon loss  (no weight) = {valid_losses[1].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid latent loss (no weight) = {valid_losses[2].item():.8f}")

    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        """
        Evaluate the validation combined loss for the model and save the model
        if a new minimum in this combined and weighted loss is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        @returns :: Pytorch loss object of the validation loss.
        """
        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        self.eval()

        latent, recon = self.forward(x_data_valid.float())

        samples = self.desired_latent_dist.sample(latent.shape)
        comparison_probs = func.softmax(samples, dim=1).to(self.device)
        latent_log_probs = func.log_softmax(latent, dim=1)

        recon_loss = self.recon_loss_function(recon, x_data_valid.float())
        laten_loss = self.laten_loss_function(latent_log_probs,comparison_probs)

        valid_loss = self.recon_loss_weight * recon_loss + \
                     self.laten_loss_weight * laten_loss
        self.save_best_loss_model(valid_loss, outdir)

        return valid_loss, recon_loss, laten_loss

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):
        """
        Train the variational autoencoder.
        @train_loader :: Pytorch data loader with the training data.
        @valid_loader :: Pytorch data loader with the validation data.
        @epochs       :: The number of epochs to train for.
        @outdir       :: The output dir where to save the training results.
        """
        self.instantiate_adam_optimizer()
        self.network_summary(); self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the " + self.hp['ae_type'] + " AE model...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()

            train_loss   = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader,outdir)
            if self.early_stopping(): break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss.append(valid_losses[1].item())
            self.all_laten_loss.append(valid_losses[2].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)
