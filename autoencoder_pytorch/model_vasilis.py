# The autoencoder used in the paper.
import numpy as np
import torch
import torch.nn as nn

class AE(nn.Module):
    # The definition of the model that Vasilis used for the paper.
    # NB: the input layer is included in the node number.
    def __init__(self, node_number, dropout=False, **kwargs):
        super(AE, self).__init__()
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

        @results :: The layers arrays of the autoencoder.
        """

        layers = []
        if encoder: layer_nbs = range(len(self.node_number))
        else: layer_nbs = reversed(range(len(self.node_number)))

        for idx in layer_nbs:
            if dropout == True:
                if idx != 0: prob=0.2; layers.append(nn.Dropout(p=prob))

            if encoder: layers.append(nn.Linear(self.node_number[idx],
                self.node_number[idx+1]))
            else:  layers.append(nn.Linear(self.node_number[idx],
                self.node_number[idx-1]))

            if encoder and idx == (len(self.node_number) - 2): break
            if (not encoder) and idx == 1: break

            layers.append(nn.Sigmoid())
            layers.append(nn.ELU(True))

        return layers

    def forward(self, x):
        # You never use this method?
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


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
        print('New minimum of validation loss: ')
        torch.save(model.state_dict(), outdir + 'best_model.pt')

    return valid_loss

def eval_train_loss(model, batch_features, criterion, optimizer, device):
    # Evaluate the training loss.
    feature_size = batch_features.shape[1]
    batch_features = batch_features.view(-1, feature_size).to(device)
    model_output,_ = model(batch_features.float())
    # Using both Sig+Bkg arrays gave problems expecting float getting double.
    train_loss = criterion(model_output, batch_features.float())
    # PyTorch accumulates gradients on subsequent backward passes.
    # Rest gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss

def train(train_loader, valid_loader, model, criterion, optimizer, epochs,
    device, outdir):
    # Train the autoencoder that was implemented above.
    print('Training the Vasilis model...')
    loss_training = []; loss_validation = []; min_valid = 99999

    for epoch in range(epochs):
        model.train()
        for i, batch_features in enumerate(train_loader):
            train_loss = \
            eval_train_loss(model,batch_features, criterion, optimizer, device)

        valid_loss = eval_valid_loss(model, valid_loader, criterion, min_valid,
            outdir, device)

        loss_validation.append(valid_loss)
        loss_training.append(train_loss.item())

        print("Epoch : {}/{}, Training loss (last batch) = {:.8f}".
            format(epoch + 1, epochs, train_loss.item()))
        print("Epoch : {}/{}, Validation loss = {:.8f}".
            format(epoch + 1, epochs, valid_loss))

    return loss_training, loss_validation, min_valid

# Are these two methods still used/useful?
def encode_array(data, saved_model, layers):
    """
    Not sure what the details of this method are. Talk to Vasilis.
    """
    data_loader = torch.utils.data.DataLoader(tensorData(data),
        batch_size=data.shape[0], shuffle = False)
    device ='cpu'
    model = AE(node_number = layers).to(device)
    model.load_state_dict(torch.load(saved_model + 'best_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        data_iter = iter(data_loader)
        input_data = data_iter.next().to(device)
        _, output = model(input_data.float())

        return output.cpu().numpy()

def encode(data, saved_model, layers):
    """
    Again, not sure what it does. Have not seen it used anywhere.
    """
    if (isinstance(data, dict)):
        for x in data:
            x = torch.Tensor(x)
            data[x] = encode_array(data[x], saved_model, layers)
        return data

    return encode_array(torch.Tensor(data),saved_model,layers)
