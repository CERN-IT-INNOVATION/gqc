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
    def __init__(self, nodes, lr, device, en_activ=None, dec_activ=None,
        **kwargs):

        super(AE, self).__init__()
        self.lr             = lr
        self.nodes          = nodes
        self.device         = device
        self.criterion      = nn.MSELoss(reduction='mean')

        self.encoder_layers = self.construct_encoder(en_activ)
        self.encoder        = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = self.construct_decoder(dec_activ)
        self.decoder        = nn.Sequential(*self.decoder_layers)

    def construct_encoder(self, en_activ):
        """
        Construct the encoder layers.
        """
        layers = []
        layer_nbs = range(len(self.nodes))
        for idx in layer_nbs:
            layers.append(nn.Linear(self.nodes[idx], self.nodes[idx+1]))
            if idx == len(self.nodes) - 2 and en_activ is None: break
            if idx == len(self.nodes) - 2: layers.append(en_activ); break
            layers.append(nn.ELU(True))

        return layers

    def construct_decoder(self, dec_activ):
        """
        Construct the decoder layers.
        """
        layers = []
        layer_nbs = reversed(range(len(self.nodes)))
        for idx in layer_nbs:
            layers.append(nn.Linear(self.nodes[idx], self.nodes[idx-1]))
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

    def eval_loss(self, init_feats, model_output):
        # Evaluate the loss function and return its value.
        return self.criterion(model_output, init_feats)

    def valid(self, valid_loader, min_valid, outdir):
        # Evaluate the validation loss for the model and save if new minimum.

        valid_data_iter = iter(valid_loader)
        valid_data      = valid_data_iter.next().to(self.device)
        self.eval()

        model_output,_ = self(valid_data.float())
        valid_loss = self.eval_loss(model_output, valid_data).item()
        if valid_loss < min_valid: min_valid = valid_loss

        if outdir is not None and min_valid == valid_loss:
            print('\033[92mNew min loss: {:.2e}\033[0m'.format(min_valid))
            torch.save(self.state_dict(), outdir + 'best_model.pt')

        return valid_loss, min_valid

    def train_batch(self, batch_feats, optimizer):
        # Train the model on a batch and evaluate the different kinds of losses.
        # Propagate this backwards for minimum train_loss.

        feature_size    = batch_feats.shape[1]
        init_feats      = batch_feats.view(-1, feature_size).to(self.device)
        model_output,_  = self(init_feats.float())

        train_loss = self.eval_loss(init_feats.float(), model_output)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        return train_loss

    def train_model(self, train_loader, valid_loader, epochs, outdir):
        print('Training the model...')
        all_train_loss = []; all_valid_loss = []; min_valid = 99999
        optimizer = self.optimizer()

        for epoch in range(epochs):
            self.train()
            for i, batch_feats in enumerate(train_loader):
                batch_train_loss = self.train_batch(batch_feats, optimizer)
            valid_loss, min_valid = self.valid(valid_loader, min_valid, outdir)

            all_valid_loss.append(valid_loss)
            all_train_loss.append(batch_train_loss.item())

            print(f"Epoch : {epoch + 1}/{epochs}, "
                  f"Training loss (last batch) = {batch_train_loss.item():.8f}")
            print(f"Epoch : {epoch + 1}/{epochs}, "
                  f"Validation loss = {valid_loss:.8f}")

        return all_train_loss, all_valid_loss, min_valid
