# Utility methods for dealing with all the different autoencoder things.
import torch
import numpy as np
import os, warnings

class tensor_data(torch.utils.data.Dataset):
    # Turn a dataset into a torch tensor dataset, that can then be passed
    # to a ML algorithm and provide training. Very needed to cast to GPU.
    def __init__(self, x):
        self.x = torch.Tensor(x)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index]

def define_torch_device():
    # Use gpu if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device, flush=True)
    return device

def to_pytorch_data(data, batch_size=None, shuffle=True):
    """
    Convert training and validation data into pytorch ready data objects.

    @data       :: The data we want to convert.
    @batch_size :: The batch size to import the data with.
    @shuffle    :: Bool of wheter to shuffle the data or not.

    @returns :: Pytorch ready data object.
    """
    if batch_size is None: batch_size = data.shape[0]
    data = tensor_data(data)
    pytorch_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
        shuffle=shuffle)

    return pytorch_loader

def split_sig_bkg(data, target):
    # Split dataset into signal and background samples using the target data.
    # The target is supposed to be 1 for every signal and 0 for every bkg.
    sig_mask   = (target == 1); bkg_mask = (target == 0)
    data_sig = data[sig_mask, :]
    data_bkg = data[bkg_mask, :]

    return data_sig, data_bkg

def load_model(model_module, layers, model_path, device):
    """
    Loads a model that was trained previously.

    @model_module :: The class of the model imported a the top.
    @layers       :: The layer structure of the model.
    @model_path   :: The path to where the trained model was saved.
    @device       :: Type of device where pytorch runs: cpu or gpu.

    @returns :: The pytorch model object as trained in the training file.
    """
    model = model_module(node_number=layers).to(device)
    model.load_state_dict(torch.load(model_path + 'best_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    return model

@torch.no_grad()
def compute_model(model, data_loader):
    """
    Computes the output of an autoencoder trained model.

    @model       :: The name of the model we are using.
    @data_loader :: Pytorch data loader object.

    @returns :: The model output, the latent space output, and the input data.
    """
    data_iter = iter(data_loader)
    input_data = data_iter.next()
    model_output, latent_output = model(input_data.float())

    return model_output, latent_output, input_data

def mean_batch_loss(data_loader, model, feature_size, criterion, device):
    # Computes the mean batch loss for a sample and model, for each batch.
    mean_loss_batch = []
    for idx, batch_features in enumerate(data_loader):
        batch_features = batch_features.view(-1, feature_size).to(device)
        output,_ = model(batch_features)
        loss = criterion(output, batch_features)
        mean_loss_batch.append(loss.item())

    return mean_loss_batch

def prepare_output(model_nodes, batch_size, learning_rate, flag):
    # Prepare the naming of training outputs. Do not print output size.
    layersTag = '.'.join(str(inode) for inode in model_nodes[1:])
    filetag = 'L' + layersTag + '_B' + str(batch_size) + '_Lr{:.0e}'.\
        format(learning_rate) + "_" + flag

    outdir = './autoencoder_pytorch/trained_models/' + filetag + '/'
    if not os.path.exists(outdir): os.makedirs(outdir)

    return filetag, outdir

def save_MSE_log(filetag, train_time, min_valid, outdir):
    # Save MSE log to check best model:
    with open(outdir + 'mse_log.txt','a+') as mse_log:
        log_entry = filetag + f': Training time = {train_time:.2f} min, Min. Validation loss = {min_valid:.6f}\n'
        mse_log.write(log_entry)

def varname(index):
    # Gets the name of what variable is currently considered based on the index
    # in the data.
    jet_feats = ["$p_t$","eta","phi","en","$p_x$","$p_y$","$p_z$","btag"]
    jet_nvars = len(jet_feats); num_jets = 7
    met_feats = ["phi","$p_t$","$p_x$","$p_y$"]
    met_nvars = len(met_feats)
    lep_feats = ["$p_t$","eta","phi","en","$p_x$","$p_y$","$p_z$"]
    lep_nvars = len(lep_feats)

    if (index < jet_nvars * num_jets):
        jet = index // jet_nvars + 1
        var = index % jet_nvars
        varstring = "Jet " + str(jet) + " " + jet_feats[var]
        return varstring

    index -= jet_nvars * num_jets;

    if (index < met_nvars):
        var = index % met_nvars;
        varstring = "MET " + met_feats[var];
        return varstring

    index -= met_nvars;

    if (index < lep_nvars):
        var = index % lep_nvars
        varstring = "Lepton " + lep_feats[var]
        return varstring;

    return None
