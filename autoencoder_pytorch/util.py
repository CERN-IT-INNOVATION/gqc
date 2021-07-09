# Utility methods for dealing with all the different autoencoder business.
import torch
import numpy as np
import os, warnings, time
import ae_vanilla
import ae_classifier
import torch.nn as nn

class tensor_data(torch.utils.data.Dataset):
    # Turn a dataset into a torch tensor dataset, that can then be passed
    # to a ML algorithm for training. Very needed to cast to GPU.
    def __init__(self, x):
        self.x = torch.Tensor(x)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index]

def choose_ae_model(user_choice, device, layers, lr, en_activ=nn.Tanh(),
    dec_activ=nn.Tanh(), class_layers=[256, 256, 128, 64, 32, 1],
    recon_weight=0.5, class_weight=0.5):
    # Picks and loads one of the implemented autencoder models.
    switcher = {
        "vanilla":   lambda : ae_vanilla_model(device, layers, lr, en_activ,
            dec_activ),
        "classifier": lambda : ae_classifier_model(device, layers, lr, en_activ,
            dec_activ, class_layers, recon_weight, class_weight)
    }
    func   = switcher.get(user_choice, lambda : "Invalid type of AE given!")
    model = func()

    return model

def ae_vanilla_model(device, layers, lr, en_activ, dec_activ):
    return ae_vanilla.AE(device, layers, lr, en_activ, dec_activ).to(device)

def ae_classifier_model(device, layers, lr, en_activ, dec_activ, class_layers,
    recon_weight, class_weight):
    return ae_classifier.AE(device, layers, lr, en_activ, dec_activ,
        class_layers, recon_weight, class_weight).to(device)

def define_torch_device():
    # Use gpu for training if available. Alert the user if not and then use cpu.
    print("\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(w): print("\033[93mGPU not available. \033[0m")


    print("\033[92mUsing device:\033[0m", device)
    return device

def to_pytorch_data(data, device, batch_size=None, shuffle=True):
    """
    Convert numpy arrays of training/validation/testing data into pytroch
    objects ready to be used in training the autoencoder.

    @data       :: 2D numpy array of the data to be converted.
    @device     :: String if the training is done on cpu or gpu.
    @batch_size :: Int of the batch size used in training.
    @shuffle    :: Bool of whether to shuffle the data or not.

    @returns :: Pytorch objects to be passed to the autoencoder for training.
    """
    if batch_size is None: batch_size = data.shape[0]
    data = tensor_data(data)
    if device == 'cpu':
        pytorch_loader = torch.utils.data.DataLoader(data,
            batch_size=batch_size, shuffle=shuffle)
    else: pytorch_loader = torch.utils.data.DataLoader(data,
            batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return pytorch_loader

def split_sig_bkg(data, target):
    # Split dataset into signal and background samples using the target data.
    # The target is supposed to be 1 for every signal and 0 for every bkg.
    sig_mask = (target == 1); bkg_mask = (target == 0)
    data_sig = data[sig_mask, :]
    data_bkg = data[bkg_mask, :]

    return data_sig, data_bkg

def load_model(model_module, model_path):
    """
    Loads a model that was trained previously.

    @model_module :: The class of the model imported a the top.
    @model_path   :: String of the path to where the trained model was saved.

    @returns :: The pytorch object of the trained autoencoder model, ready to
        use for encoding/decoding data.
    """
    model.load_state_dict(torch.load(model_path + 'best_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    return model

def extract_batch_from_model_path(model_path):
    # Extract the batch size information from the folder name of a trained
    # model and return it.
    start_idx = model_path.find("_B") + 2
    end_idx = model_path[start_idx:].find("_")

    return int(model_path[batch_idx:end_idx])

def extract_layers_from_model_path(model_path):
    # Extract the layer structure information from the folder name of a trained
    # model and return it.
    start_idx = model_path.find("L") + 1
    end_idx = model_path[start_idx:].find("_")

    return model_path[batch_idx:end_idx]

@torch.no_grad()
def compute_model(model, data_loader):
    """
    Computes the output of an autoencoder trained model.

    @model       :: The name of the model we are using.
    @data_loader :: Pytorch data loader obj containing data to give the model.

    @returns :: The decoder out, the latent space, and the input data.
    """
    data_iter = iter(data_loader)
    input_data = data_iter.next()
    model_output, latent_output = model(input_data.float())

    return model_output, latent_output, input_data

def prep_out(model, batch_size, learning_rate, maxdata, flag):
    # Create the folder for the output of the model training.
    # Save the model architecture to a text file inside that folder.
    layers_tag = '.'.join(str(inode) for inode in model.layers[1:])
    file_tag   = 'L' + layers_tag + '_B' + str(batch_size) + \
        f'_Lr{learning_rate:.0e}' + "_" + f"data{maxdata:.2e}" + "_" + flag

    outdir = './trained_models/' + file_tag + '/'
    if not os.path.exists(outdir): os.makedirs(outdir)
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
       print(model, file=model_architecture)
       print(model, file=model_architecture)

    return outdir

def varname(index):
    # Gets the name of what variable is currently considered based on the index
    # in the data array.
    jet_feats=["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$",
        "btag"]
    jet_nvars=len(jet_feats); num_jets = 7
    met_feats=["$\\phi$","$p_t$","$p_x$","$p_y$"]
    met_nvars=len(met_feats)
    lep_feats=["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$"]
    lep_nvars=len(lep_feats)

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
