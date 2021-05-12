# The autoencoder used in the paper.
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
    def __init__(self, node_number, lr, dropout=False, **kwargs):
        super(AE, self).__init__()
        self.lr = lr
        self.node_number = node_number

        self.encoder_layers = self.construct_layers(True, dropout)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = self.construct_layers(False, dropout)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def construct_layers(self, encoder, dropout=False):
        """
        Construct the layers of the autoencoder module, wheter decoder
        or encoder.

        @encoder :: Bool, true if encoder, false if decoder.
        @dropout :: Drop any node in a layer with 20% probability.

        @results :: The layers arrays of the autoencoder.
        """

        layers = []; decoder = not encoder
        if encoder: layer_nbs = range(len(self.node_number))
        if decoder: layer_nbs = reversed(range(len(self.node_number)))

        for idx in layer_nbs:
            if dropout == True:
                if idx != 0: prob=0.2; layers.append(nn.Dropout(p=prob))

            if encoder: layers.append(nn.Linear(self.node_number[idx],
                self.node_number[idx+1]))
            else:  layers.append(nn.Linear(self.node_number[idx],
                self.node_number[idx-1]))

            encoder_last = idx != (len(self.node_number) - 2)
            decoder_last = idx != 1
            if encoder and encoder_last: layers.append(nn.ELU(True)); continue
            if decoder and decoder_last: layers.append(nn.ELU(True)); continue
            layers.append(nn.Sigmoid()); break

        return layers

    def forward(self, x):
        # Need to define it even if you do not use it.
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def criterion(self): return nn.MSELoss(reduction='mean')
    def optimizer(self): return optim.Adam(self.parameters(), lr=self.lr)

def eval_valid_loss(model, valid_loader, criterion, min_valid, outdir, device):
    # Evaluate the model using the validation data. Compute the loss.
    # If the new loss is a minimum, save this as the best model.
    valid_data_iter = iter(valid_loader)
    valid_data = valid_data_iter.next().to(device)
    model.eval()

    model_output,_ = model(valid_data.float())
    valid_loss = criterion(model_output, valid_data).item()
    if valid_loss < min_valid:
        min_valid = valid_loss
        if outdir is not None:
            print('New min of loss: {:.2e}'.format(min_valid), flush=True)
            torch.save(model.state_dict(), outdir + 'best_model.pt')

    return valid_loss, min_valid

def eval_train_loss(model, batch_features, criterion, optimizer, device):
    # Evaluate the training loss.
    feature_size   = batch_features.shape[1]
    batch_features = batch_features.view(-1, feature_size).to(device)
    model_output,_ = model(batch_features.float())
    train_loss = criterion(model_output, batch_features.float())
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss, optimizer

def train(train_loader, valid_loader, model, device, epochs, outdir):
    # Train the autoencoder that was implemented above.
    print('Training the Vasilis model...', flush=True)
    loss_training = []; loss_validation = []; min_valid = 99999
    optimizer = model.optimizer()
    criterion = model.criterion()

    for epoch in range(epochs):
        model.train()
        for i, batch_features in enumerate(train_loader):
            train_loss, optimizer = eval_train_loss(model, batch_features,
                criterion, optimizer, device)
        valid_loss, min_valid = eval_valid_loss(model, valid_loader, criterion,
            min_valid, outdir, device)
        loss_validation.append(valid_loss)
        loss_training.append(train_loss.item())
        print("Epoch : {}/{}, Training loss (last batch) = {:.8f}".
               format(epoch + 1, epochs, train_loss.item()), flush=True)
        print("Epoch : {}/{}, Validation loss = {:.8f}".
               format(epoch + 1, epochs, valid_loss), flush=True)

    return loss_training, loss_validation, min_valid
