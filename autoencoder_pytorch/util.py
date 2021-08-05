# Utility methods for dealing with all the different autoencoder business.
import torch
import torch.nn as nn

import numpy as np
import os, warnings, time, json

import ae_vanilla
import ae_classifier
import ae_svm
from terminal_colors import tcols

def choose_ae_model(ae_type, device, hyperparams):
    # Picks and loads one of the implemented autencoder models.
    switcher = {
        "vanilla": lambda: ae_vanilla.AE_vanilla(device, hyperparams),
        "classifier": lambda: ae_classifier.AE_classifier(device, hyperparams),
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
    # Import hyperparameters from json file that stores them.
    file_path = os.path.join(model_path, "hyperparameters.json")
    hyperparams_file = open(file_path,)
    hyperparams = json.load(hyperparams_file)
    hyperparams_file.close()

    return hyperparams

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
