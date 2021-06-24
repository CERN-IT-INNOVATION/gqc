# The autoencoder architecture that Vasilis started with. Reduces the number
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
    # The definition of the model that Vasilis used for the paper.
    # NB: the input layer is included in the node number.
    def __init__(self, nodes, lr, device, en_activ=None, dec_activ=None,
        **kwargs):

        super(AE, self).__init__()
        self.lr = lr
        self.nodes  = nodes
        self.device = device

        self.encoder_layers = self.construct_encoder(en_activ)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = self.construct_decoder(dec_activ)
        self.decoder = nn.Sequential(*self.decoder_layers)

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

    def get_dev(self):   return self.device
    def criterion(self): return nn.MSELoss(reduction='mean')
    def optimizer(self): return optim.Adam(self.parameters(), lr=self.lr)

    def eval_criterion(self, init_feats, model_output):
        # Evaluate the loss function and return its value.
        criterion = self.criterion()
        loss_recons = criterion(model_output, init_feats)

        return loss_recons

def eval_valid(model, valid_loader, min_valid, outdir):
    # Evaluate the validation loss for the model and save if got new minimum.
    valid_data_iter = iter(valid_loader)
    valid_data = valid_data_iter.next().to(model.get_dev())
    model.eval()

    model_output,_ = model(valid_data.float())
    valid_loss = model.eval_criterion(model_output, valid_data).item()
    if valid_loss < min_valid:
        min_valid = valid_loss

    if outdir is not None and min_valid == valid_loss:
        print('\033[92mNew min loss: {:.2e}\033[0m'.format(min_valid))
        torch.save(model.state_dict(), outdir + 'best_model.pt')

    return valid_loss, min_valid

def eval_train(model, batch_feats, optimizer):
    # Evaluate the training loss.
    feature_size  = batch_feats.shape[1]
    init_feats    = batch_feats.view(-1, feature_size).to(model.get_dev())
    model_output,_  = model(init_feats.float())

    train_loss = model.eval_criterion(init_feats.float(), model_output)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss

def train(train_loader, valid_loader, model, epochs, outdir):
    # Training method for the autoencoder that was defined above.
    print('Training the model...')
    loss_training = []; loss_validation = []; min_valid = 99999
    optimizer = model.optimizer()

    for epoch in range(epochs):
        model.train()
        for i, batch_feats in enumerate(train_loader):
            train_loss = eval_train(model, batch_feats, optimizer)
        valid_loss, min_valid = eval_valid(model, valid_loader, min_valid,
            outdir)

        loss_validation.append(valid_loss)
        loss_training.append(train_loss.item())
        print("Epoch : {}/{}, Training loss (last batch) = {:.8f}".
               format(epoch + 1, epochs, train_loss.item()))
        print("Epoch : {}/{}, Validation loss = {:.8f}".
               format(epoch + 1, epochs, valid_loss))

    return loss_training, loss_validation, min_valid
