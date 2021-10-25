# Utility methods for dealing with all the different autoencoder business.
import torch
import torch.nn as nn

import numpy as np
import os, warnings, time, json

from ae_vanilla import AE_vanilla
from ae_classifier import AE_classifier
from ae_variational import AE_variational
from ae_sinkhorn import AE_sinkhorn
from ae_vqc import AE_vqc
from ae_sinkclass import AE_sinkclass

from terminal_colors import tcols

def choose_ae_model(ae_type, device, hyperparams):
    """
    Picks and loads one of the implemented autoencoder models.
    @ae_type     :: String of the type of autoencoder that you want to load.
    @device      :: String of the device to load it on: 'cpu' or 'gpu'.
    @hyperparams :: Dictionary of the hyperparameters to load with.

    @returns :: The loaded autoencoder model with the given hyperparams.
    """
    switcher = {
        "vanilla"    : lambda: AE_vanilla(device, hyperparams),
        "classifier" : lambda: AE_classifier(device, hyperparams),
        "variational": lambda: AE_variational(device, hyperparams),
        "sinkhorn"   : lambda: AE_sinkhorn(device, hyperparams),
        "classvqc"   : lambda: AE_vqc(device, hyperparams),
        "sinkclass"  : lambda: AE_sinkclass(device, hyperparams),
    }
    model = switcher.get(ae_type, lambda: None)()
    if model is None: raise TypeError("Specified AE type does not exist!")

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

def import_hyperparams(model_path):
    """
    Import hyperparameters from json file that stores them.
    @model_path :: String of the path to a trained pytorch model folder to
         import hyperparameters from the json file inside that folder.

    @returns :: Imported dictionary of hyperparams from .json file inside the
        trained model folder.
    """
    file_path = os.path.join(model_path, "hyperparameters.json")
    hyperparams_file = open(file_path,)
    hyperparams = json.load(hyperparams_file)
    hyperparams_file.close()

    return hyperparams

def varname(index):
    # Gets the name of what variable is currently considered based on the index
    # in the data array.
    # I really hate this method, need to find a better way of assigning names.

    jet_feat = ["$p_T$","$\\eta$","$\\phi$", "E","$p_x$","$p_y$","$p_z$","btag"]
    jet_nvar = len(jet_feat); num_jets = 7
    met_feat = ["$\\phi$","$p_t$","$p_x$","$p_y$"]
    met_nvar = len(met_feat)
    lep_feat = ["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$"]
    lep_nvar = len(lep_feat)

    if (index < jet_nvar * num_jets):
        jet = index // jet_nvar + 1
        var = index % jet_nvar
        varstring = "Jet " + str(jet) + " " + jet_feat[var]
        return varstring
    index -= jet_nvar * num_jets;

    if (index < met_nvar):
        var = index % met_nvar;
        varstring = "MET " + met_feat[var];
        return varstring
    index -= met_nvar;

    if (index < lep_nvar):
        var = index % lep_nvar
        varstring = "Lepton " + lep_feat[var]
        return varstring;

    return None
