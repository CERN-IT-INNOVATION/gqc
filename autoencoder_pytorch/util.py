# Utility methods for dealing with all the different autoencoder business.
import torch
import torch.nn as nn

import numpy as np
import os, warnings, time

import ae_vanilla
import ae_classifier
import ae_svm
from terminal_colors import tcols

def choose_ae_model(user_choice, device, layers, lr, en_activ=nn.Tanh(),
    dec_activ=nn.Tanh(), class_layers=[256, 256, 128, 64, 32, 1],
    loss_weight=0.5):
    # Picks and loads one of the implemented autencoder models.
    switcher = {
        "vanilla": ae_vanilla.AE_vanilla(device,layers,lr,en_activ,dec_activ),
        "classifier": ae_classifier.AE_classifier(device, layers, lr, en_activ,
            dec_activ, class_layers, loss_weight),
        "svm": ae_svm.AE_svm(device, layers, lr, en_activ, dec_activ,
            loss_weight)
    }
    model = switcher.get(user_choice)

    return model

def define_torch_device():
    # Use gpu for training if available. Alert the user if not and then use cpu.
    print("\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(w): print(tcols.WARNING + "GPU not available." + tcols.ENDC)


    print("\033[92mUsing device:\033[0m", device)
    return device

def get_train_file(norm_name, nevents):
    return "x_data_" + norm_name + "_norm_" + nevents + "_train.npy"

def get_valid_file(norm_name, nevents):
    return "x_data_" + norm_name + "_norm_" + nevents + "_valid.npy"

def get_test_file(norm_name, nevents):
    return "x_data_" + norm_name + "_norm_" + nevents + "_test.npy"

def get_train_target(norm_name, nevents):
    return "y_data_" + norm_name + "_norm_" + nevents + "_train.npy"

def get_valid_target(norm_name, nevents):
    return "y_data_" + norm_name + "_norm_" + nevents + "_valid.npy"

def get_test_target(norm_name, nevents):
    return "y_data_" + norm_name + "_norm_" + nevents + "_test.npy"

def to_pytorch_data(data, target, device, batch_size=None, shuffle=True):
    """
    Convert numpy arrays of training/validation/testing data into pytroch
    objects ready to be used in training the autoencoder.

    @data       :: 2D numpy array of the data to be converted.
    @target     :: 1D numpy array of the target dataset.
    @device     :: String if the training is done on cpu or gpu.
    @batch_size :: Int of the batch size used in training.
    @shuffle    :: Bool of whether to shuffle the data or not.

    @returns :: Pytorch objects to be passed to the autoencoder for training.
    """
    if batch_size is None: batch_size = data.shape[0]
    data    = torch.Tensor(data)
    target  = torch.Tensor(target)
    dataset = torch.utils.data.TensorDataset(data, target)

    if device == 'cpu':
        pytorch_loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=shuffle)
    else:
        pytorch_loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return pytorch_loader

def split_sig_bkg(data, target):
    # Split dataset into signal and background samples using the target data.
    # The target is supposed to be 1 for every signal and 0 for every bkg.
    sig_mask = (target == 1); bkg_mask = (target == 0)
    data_sig = data[sig_mask, :]
    data_bkg = data[bkg_mask, :]

    return data_sig, data_bkg

def load_model(model, model_path):
    # Loads a pytorch saved model.pt file given it's path.
    if not os.path.exists(model_path): raise FileNotFoundError("∄ path.")
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu')))

    return model

def import_hyperparams(model_path):
    # Import the hyperparameters given the path to the model, since the name
    # of the model folder has the layers, batch, and learning rate in it.

    hyperparams = max(model_path.split('/'), key=len)
    layers  = hyperparams[hyperparams.find('L')+1:hyperparams.find('_')]
    layers  = [int(nb) for nb in layers.split(".")]
    batch   = int(hyperparams[hyperparams.find('B')+1:hyperparams.find('_',
        hyperparams.find('B')+1, len(hyperparams))])
    lr      = float(hyperparams[hyperparams.find('Lr')+2:hyperparams.find('_',
        hyperparams.find('Lr')+2, len(hyperparams))])
    nevents = hyperparams[hyperparams.find('N')+1:hyperparams.find('_',
        hyperparams.find('N')+1, len(hyperparams))]
    norm    = hyperparams[hyperparams.find('S')+1:hyperparams.find('_',
        hyperparams.find('S')+1, len(hyperparams))]
    aetype  = hyperparams[hyperparams.find('T')+1:hyperparams.find('_',
        hyperparams.find('T')+1, len(hyperparams))]

    print(tcols.OKGREEN + "\nImported model hyperparameters:" + tcols.ENDC)
    print("--------------------------------")
    print(f"Layers: {layers}")
    print(f"Batch: {batch}")
    print(f"Learning Rate: {lr}")
    print(f"Normalisation Name: {norm}")
    print(f"Number of Events: {nevents}")

    return layers, aetype, batch, lr, norm, nevents

def prep_out(model, aetype, batch_size, learning_rate, maxdata, norm, flag):
    # Create the folder for the output of the model training.
    # Save the model architecture to a text file inside that folder.
    layers_tag = '.'.join(str(inode) for inode in model.layers[1:])
    file_tag   = 'L' + layers_tag + '_T' + aetype + '_B' + str(batch_size) + \
        f'_Lr{learning_rate:.0e}' + "_" + f"N{maxdata}" + "_S" + norm + \
        "_" + flag

    outdir = './trained_models/' + file_tag + '/'
    if not os.path.exists(outdir): os.makedirs(outdir)
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
       print(model, file=model_architecture)

    return outdir

def varname(index):
    # Gets the name of what variable is currently considered based on the index
    # in the data array.
    jet_feats = ["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$",
        "btag"]
    jet_nvars = len(jet_feats); num_jets = 7
    met_feats = ["$\\phi$","$p_t$","$p_x$","$p_y$"]
    met_nvars = len(met_feats)
    lep_feats = ["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$"]
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
