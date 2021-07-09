# The vanilla autoencoder architecture. Reduces the number
# of features from 67 down to 16 by using a combination of linear and ELU
# layers. The loss function of the autoencoder is the MSE between the
# histograms reconstructed by the decoder and the original variables.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classifiers import FFWD
seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE(nn.Module):
    def __init__(self, device, layers, lr, en_activ, dec_activ, class_layers,
        recon_weight, class_weight, **kwargs):

        super(AE, self).__init__()
        self.lr                  = lr
        self.layers              = layers
        self.device              = device
        self.recon_loss_function = nn.MSELoss(reduction='mean')
        self.class_loss_function = nn.BCELoss()
        self.recon_weight        = recon_weight
        self.class_weight        = class_weight

        (class_layers).insert(0, layers[0])
        self.classifier = FFWD(device, class_layers)

        self.encoder_layers = self.construct_encoder(en_activ)
        self.encoder        = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = self.construct_decoder(dec_activ)
        self.decoder        = nn.Sequential(*self.decoder_layers)

    def construct_encoder(self, en_activ):
        """
        Construct the encoder layers.
        """
        layers = []
        layer_nbs = range(len(self.layers))
        for idx in layer_nbs:
            layers.append(nn.Linear(self.layers[idx], self.layers[idx+1]))
            if idx == len(self.layers) - 2 and en_activ is None: break
            if idx == len(self.layers) - 2: layers.append(en_activ); break
            layers.append(nn.ELU(True))

        return layers

    def construct_decoder(self, dec_activ):
        """
        Construct the decoder layers.
        """
        layers = []
        layer_nbs = reversed(range(len(self.layers)))
        for idx in layer_nbs:
            layers.append(nn.Linear(self.layers[idx], self.layers[idx-1]))
            if idx == 1 and dec_activ is None: break
            if idx == 1 and dec_activ: layers.append(dec_activ); break
            layers.append(nn.ELU(True))

        return layers

    def forward(self, x):
        # Need to define it even if you do not use it.
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def optimizer(self): return optim.Adam(self.parameters(), lr=self.lr)

    def recon_loss(self, init_feats, model_output):
        return self.recon_loss_function(model_output, init_feats)

    def class_loss(self, x_train, y_train):
        return self.classifier.fit(x_train, y_train)

    def total_loss(self, recon_loss, class_loss):
        return self.recon_weight*recon_loss + self.class_weight*class_loss

    def valid(self, valid_loader, valid_target, min_valid, outdir):
        # Evaluate the validation loss for the model and save if new minimum.

        valid_data_iter = iter(valid_loader)
        valid_data      = valid_data_iter.next().to(self.device)
        self.eval()

        model_output, latent_output = self(valid_data.float())
        recon_loss = self.recon_loss(valid_data.float(), model_output)
        class_loss = self.class_loss(latent_output.numpy(), valid_target)
        valid_loss = self.total_loss(recon_loss, class_loss)

        if valid_loss < min_valid: min_valid = valid_loss

        if outdir is not None and min_valid == valid_loss:
            print('\033[92mNew min loss: {:.2e}\033[0m'.format(min_valid))
            torch.save(self.state_dict(), outdir + 'best_model.pt')

        return valid_loss, min_valid

    def train_batch(self, x_batch, y_batch, optimizer):
        # Train the model on a batch and evaluate the different kinds of losses.
        # Propagate this backwards for minimum train_loss.

        feature_size = x_batch.shape[1]
        init_feats   = x_batch.view(-1, feature_size).to(self.device)
        model_output, latent_output = self(init_feats.float())

        recon_loss = self.recon_loss(init_feats.float(), model_output)
        class_loss = self.class_loss(latent_output.numpy(), y_batch)
        train_loss = self.total_loss(recon_loss, class_loss)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        return train_loss

    def train_all_batches(self, train_loader, train_target, optimizer):

        for idx, x_batch in enumerate(train_loader):
            y_batch = train_target[idx*len(x_batch):(idx+1)*len(x_batch)]
            batch_train_loss = self.train_batch(x_batch, y_batch, optimizer)

        return batch_train_loss

    def train_model(self, train_loader, valid_loader, train_target,
        valid_target, epochs, outdir):

        print('\033[96mTraining the classifier AE model...\033[0m')
        all_train_loss = []
        all_valid_loss = []
        min_valid      = 99999
        optimizer      = self.optimizer()

        for epoch in range(epochs):
            self.train()
            last_batch_train_loss = \
                self.train_all_batches(train_loader, train_target, optimizer)
            valid_loss, min_valid = \
                self.valid(valid_loader, valid_target, min_valid, outdir)

            all_valid_loss.append(valid_loss)
            all_train_loss.append(batch_train_loss.item())

            print(f"Epoch : {epoch + 1}/{epochs}, "
                  f"Training loss (last batch) = {batch_train_loss.item():.8f}")
            print(f"Epoch : {epoch + 1}/{epochs}, "
                  f"Validation loss = {valid_loss:.8f}")

        return all_train_loss, all_valid_loss, min_valid
