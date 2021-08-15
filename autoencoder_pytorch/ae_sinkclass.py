# The end-to-end Sinkhorn autoencoder. For more details, please see the
# publication https://arxiv.org/abs/2006.06704.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import geomloss

from ae_vanilla import AE_vanilla
from terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_sinkclass(AE_vanilla):
    def __init__(self, device='cpu', hparams={}):

        super().__init__(device, hparams)
        new_hp = {
            "ae_type"                : "sinkclass",
            "noise_gen_input_layers" : self.hp["ae_layers"][:2],
            "class_layers"           : [128, 64, 32, 16, 8, 1],
            "labels_dimension"       : 2,
            "adam_betas"             : (0.9, 0.999),
            "loss_weight"            : 0.5,
            "weight_sink"            : 1
        }

        self.hp.update(new_hp)
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())

        self.class_loss_weight = self.hp['loss_weight']
        self.laten_loss_weight = self.hp['weight_sink']

        self.class_loss_function = nn.BCELoss(reduction='mean')
        self.laten_loss_function = geomloss.SamplesLoss("sinkhorn", blur=0.05,
            scaling=0.95, diameter=0.01, debias=True)

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.construct_noise_gen_input()
        self.construct_noise_generator()
        self.class_layers = [self.hp['ae_layers'][-1]] + self.hp['class_layers']
        self.classifier   = self.construct_classifier(self.class_layers)

        self.all_recon_loss = []
        self.all_laten_loss = []
        self.all_class_loss = []

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

    def construct_noise_gen_input(self):
        """
        Construct the input to the noise generator, which is not trivial.
        """
        input_layers = self.hp["noise_gen_input_layers"]
        labels_dim   = self.hp["labels_dimension"]

        self.noise_gen_input_data = nn.Sequential(
            nn.Linear(input_layers[0], input_layers[1]),   nn.LeakyReLU(0.2),
            nn.Linear(input_layers[1], input_layers[1]*2), nn.LeakyReLU(0.2))
        self.noise_gen_input_labl = nn.Sequential(
            nn.Linear(labels_dim, input_layers[1]), nn.LeakyReLU(0.2))

    def construct_noise_generator(self):
        """
        Construct the noise generator layers. Make sure that the layer is
        Linear for input_dim to work. Otherwise... Look at pt documentation.
        """
        noise_gen_layers = []
        input_dim = self.noise_gen_input_data[2].in_features
        layers    = [input_dim*3, input_dim*4, input_dim*3, int(input_dim/4)]
        layer_nbs = range(len(layers))
        for idx in layer_nbs:
            noise_gen_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            if idx == len(layers) - 2: break
            noise_gen_layers.append(nn.LeakyReLU(0.2))

        self.noise_gen = nn.Sequential(*noise_gen_layers)

    def generate_noise(self, x, y):
        # Use the noise generator nn to generate noise.
        x_data = self.noise_gen_input_data(x)
        y_data = self.noise_gen_input_labl(y)

        xy_conc = torch.cat((x_data, y_data), 1)

        xy_outp = self.noise_gen(xy_conc)

        return xy_outp

    def transform_target_data(self, y_batch):
        batch_size = y_batch.shape[0]
        y_map   = torch.FloatTensor(batch_size, self.hp["labels_dimension"])
        y_map   = y_map.to(self.device)
        y_batch = y_batch.to(self.device)
        y_map.zero_()

        return y_map.scatter_(1, y_batch.reshape([-1, 1]).type(torch.int64), 1)\
               .to(self.device)

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

        y_data_trans = self.transform_target_data(y_data)
        latent, classif, recon = self.forward(x_data.float())

        noise_input_probs = torch.rand(x_data.shape).to(self.device)
        latent_noise      = self.generate_noise(noise_input_probs, y_data_trans)
        latent_noise      = torch.cat([latent_noise, y_data_trans], 1)
        latent            = torch.cat([latent, y_data_trans], 1)

        class_loss = self.class_loss_function(classif.flatten(), y_data)
        laten_loss = self.laten_loss_function(latent, latent_noise)
        recon_loss = self.recon_loss_function(recon, x_data.float())

        return recon_loss + \
               self.laten_loss_weight*laten_loss + \
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
              f"Valid sinkh loss (no weight) = {valid_losses[2].item():.8f}")
        print(f"Epoch : {epoch + 1}/{epochs}, "
              f"Valid class loss (no weight) = {valid_losses[3].item():.8f}")

    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        # Evaluate the validation loss for the model and save if new minimum.

        x_data_valid, y_data_valid = iter(valid_loader).next()
        x_data_valid = x_data_valid.to(self.device)
        y_data_valid = y_data_valid.to(self.device)
        y_data_trans = self.transform_target_data(y_data_valid)
        self.eval()

        latent, classif, recon = self.forward(x_data_valid.float())

        noise_input_probs = torch.rand(x_data_valid.shape).to(self.device)
        latent_noise      = self.generate_noise(noise_input_probs, y_data_trans)
        latent_noise      = torch.cat([latent_noise, y_data_trans], 1)
        latent            = torch.cat([latent, y_data_trans], 1)

        class_loss = self.class_loss_function(classif.flatten(), y_data_valid)
        recon_loss = self.recon_loss_function(recon, x_data_valid.float())
        laten_loss = self.laten_loss_function(latent, latent_noise)

        valid_loss = recon_loss + \
                     self.laten_loss_weight*laten_loss + \
                     self.class_loss_weight*class_loss

        self.save_best_loss_model(valid_loss, outdir)

        return valid_loss, recon_loss, laten_loss, class_loss

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

        for epoch in range(epochs):
            self.train()

            train_loss   = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader,outdir)
            if self.early_stopping(): break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss.append(valid_losses[1].item())
            self.all_laten_loss.append(valid_losses[2].item())
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
