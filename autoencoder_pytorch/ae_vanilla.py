# The vanilla autoencoder architecture. Reduces the number
# of features from 67 down to 16. The loss function of the autoencoder
# is the MSE between the data reconstructed by the decoder and the original.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import os, json
import matplotlib.pyplot as plt

from .terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need. Disabled to increase performance.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_vanilla(nn.Module):
    def __init__(self, device='cpu', hparams={}):

        super().__init__()
        self.hp = {
            "ae_type"    : "vanilla",
            "ae_layers"  : [67, 64, 52, 44, 32, 24, 16],
            "lr"         : 0.002,
            "enc_activ"  : 'nn.Tanh()',
            "dec_activ"  : 'nn.Tanh()',
        }
        self.device    = device

        self.recon_loss_function = nn.MSELoss(reduction='mean')
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())

        exec('self.enc_activ = ' + self.hp["enc_activ"])
        exec('self.dec_activ = ' + self.hp["dec_activ"])

        self.best_valid_loss = 9999
        self.all_train_loss  = []
        self.all_valid_loss  = []

        self.early_stopping_limit = 15
        self.epochs_no_improve    = 0

        self.encoder = \
            self.construct_encoder(self.hp['ae_layers'], self.enc_activ)
        self.decoder = \
            self.construct_decoder(self.hp['ae_layers'], self.dec_activ)

    @staticmethod
    def construct_encoder(layers, enc_activ):
        """
        Construct the encoder.
        @layers    :: Array of number of nodes for each layer.
        @enc_activ :: Pytorch object for encoder activation function.

        @returns  :: Pytorch sequence of layers making the encoder.
        """
        enc_layers = []
        layer_nbs = range(len(layers))
        for idx in layer_nbs:
            enc_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            if idx == len(layers) - 2 and enc_activ is None: break
            if idx == len(layers) - 2: enc_layers.append(enc_activ); break
            enc_layers.append(nn.ReLU(True))

        return nn.Sequential(*enc_layers)

    @staticmethod
    def construct_decoder(layers, dec_activ):
        """
        Construct the decoder.
        @layers   :: Array of number of nodes for each layer.
        @dec_activ :: Pytorch object for decoder activation function.

        @returns  :: Pytorch sequence of layers making the decoder.
        """
        dec_layers = []
        layer_nbs = reversed(range(len(layers)))
        for idx in layer_nbs:
            dec_layers.append(nn.Linear(layers[idx], layers[idx-1]))
            if idx == 1 and dec_activ is None: break
            if idx == 1 and dec_activ: dec_layers.append(dec_activ); break
            dec_layers.append(nn.ELU(True))

        return nn.Sequential(*dec_layers)

    def instantiate_adam_optimizer(self):
        """
        Instantiate the optimizer object, used in the training of the ae.
        Also add exponential learning rate decay that can be used in the
        training method (optional).
        """
        self = self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.hp['lr'])
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def forward(self, x):
        """
        Forward pass through the ae.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def compute_loss(self, x_data):
        """
        Compute the loss of a forward pass through the ae.
        @x_data  :: Numpy array of the original input data.

        @returns :: Float of the computed loss function value.
        """
        if type(x_data) is np.ndarray:
            x_data = torch.from_numpy(x_data).to(self.device)
        latent, recon = self.forward(x_data.float())
        return self.recon_loss_function(recon, x_data.float())

    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_loss):
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
              f"Valid loss = {valid_loss.item():.8f}")

    @staticmethod
    def print_summary(model):
        """
        Prints a neat summary of a given ae model, with all the layers.
        @model :: Pytorch object of the model to be printed.
        """
        try: summary(model, show_input=True, show_hierarchical=False,
                print_summary=True, max_depth=1, show_parent_layers=False)
        except: print(tcols.WARNING + "Net summary failed!" + tcols.ENDC)

    def network_summary(self):
        """
        Prints a summary of the entire ae network.
        """
        print(tcols.OKGREEN + "Encoder summary:" + tcols.ENDC)
        self.print_summary(self.encoder)
        print('\n')
        print(tcols.OKGREEN + "Decoder summary:" + tcols.ENDC)
        self.print_summary(self.decoder)
        print('\n\n')

    def optimizer_summary(self):
        """
        Prints a summary of the optimizer that is used in the training.
        """
        print(tcols.OKGREEN + "Optimizer summary:" + tcols.ENDC)
        print(self.optimizer)
        print('\n\n')

    def save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimised model with the best loss.
        @valid_loss :: Float of the validation loss.
        @outdir     :: Directory where the best model is saved.
        """
        if self.best_valid_loss > valid_loss:
            self.epochs_no_improve = 0
            print(tcols.OKGREEN + f"New min: {self.best_valid_loss:.2e}" +
                  tcols.ENDC)

            self.best_valid_loss = valid_loss
            if not outdir is None:
                torch.save(self.state_dict(), outdir + 'best_model.pt')

        else: self.epochs_no_improve += 1

    def early_stopping(self):
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.
        """
        if self.epochs_no_improve >= self.early_stopping_limit:
            return 1
        return 0

    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        """
        Evaluate the validation loss for the model and save the model if a
        new minimum is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        @returns :: Pytorch loss object of the validation loss.
        """
        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        self.eval()

        loss = self.compute_loss(x_data_valid, None)
        self.save_best_loss_model(loss, outdir)

        return loss

    def train_batch(self, x_batch, y_batch=None):
        """
        Train the model on a batch and evaluate the different kinds of losses.
        Propagate this backwards for minimum train_loss.
        @x_batch :: Pytorch batch object with the data.
        @y_batch :: Pytorch batch object with the target.

        @returns :: Pytorch loss object of the training loss.
        """
        feature_size  = x_batch.shape[1]
        init_feats    = x_batch.view(-1,feature_size).to(self.device)
        if not y_batch is None: y_batch = y_batch.to(self.device)

        loss = self.compute_loss(init_feats, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_all_batches(self, train_loader):
        """
        Train the autoencoder on all the batches.
        @train_loader :: Pytorch loader object with the training data.

        @returns :: The normalised training loss over all the batches.
        """
        batch_loss_sum = 0
        nb_of_batches  = 0
        for batch in train_loader:
            x_batch, y_batch = batch
            batch_loss       = self.train_batch(x_batch)
            batch_loss_sum   += batch_loss
            nb_of_batches    += 1

        return batch_loss_sum/nb_of_batches

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):
        """
        Train the vanilla autoencoder.
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

            train_loss = self.train_all_batches(train_loader)
            valid_loss = self.valid(valid_loader, outdir)
            if self.early_stopping(): break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_loss.item())
            self.print_losses(epoch, epochs, train_loss, valid_loss)

    def loss_plot(self, outdir):
        """
        Plots the loss for each epoch for the training and validation data.
        @outdir :: Directory where to save the loss plot.
        """
        epochs = list(range(len(self.all_train_loss)))
        plt.plot(epochs, self.all_train_loss,
            color="gray", label="Training Loss (average)")
        plt.plot(epochs, self.all_valid_loss,
            color="navy", label="Validation Loss")
        plt.xlabel("Epochs"); plt.ylabel("Loss")

        plt.text(np.min(epochs), np.max(self.all_train_loss),
            f"Min: {self.best_valid_loss:.2e}", verticalalignment='top',
            horizontalalignment='left', color='blue', fontsize=15,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

        plt.legend()
        plt.savefig(outdir + "loss_epochs.pdf")
        plt.close()

        print(tcols.OKGREEN + f"Loss vs epochs plot saved to {outdir}." +
              tcols.ENDC)

    def export_architecture(self, outdir):
        """
        Saves the structure of the nn to a file.
        @outdir :: Directory where to save the architecture of the network.
        """
        with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
            print(self, file=model_architecture)

    def export_hyperparameters(self, outdir):
        """
        Saves the hyperparameters of the model to a json file.
        @outdir :: Directory where to save the json file.
        """
        file_path   = os.path.join(outdir, 'hyperparameters.json')
        params_file = open(file_path, 'w')
        json.dump(self.hp, params_file)
        params_file.close()

    def load_model(self, model_path):
        """
        Loads the weights of a trained model saved in a .pt file.
        @model_pathr :: Directory where a trained model was saved.
        """
        model_path = os.path.join(model_path, 'best_model.pt')
        if not os.path.exists(model_path): raise FileNotFoundError("∄ path.")
        self.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device)))

    @torch.no_grad()
    def predict(self, x_data):
        """
        Compute the prediction of the autoencoder.
        @x_data :: Input array to pass through the autoencoder.
        """
        x_data = torch.from_numpy(x_data).to(self.device)
        self.eval()
        latent, reconstructed = self.forward(x_data.float())

        latent        = latent.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        return latent, reconstructed
