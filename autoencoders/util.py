# Utility methods for dealing with all the different autoencoder business.
import torch

import warnings
import json
import subprocess

from .ae_vanilla import AE_vanilla
from .ae_classifier import AE_classifier
from .ae_variational import AE_variational
from .ae_sinkhorn import AE_sinkhorn
from .ae_sinkclass import AE_sinkclass

from .terminal_colors import tcols


def choose_ae_model(ae_type, device, hyperparams) -> callable:
    """
    Picks and loads one of the implemented autoencoder model classes.
    @ae_type     :: String of the type of autoencoder that you want to load.
    @device      :: String of the device to load it on: 'cpu' or 'gpu'.
    @hyperparams :: Dictionary of the hyperparameters to load with.

    returns :: The loaded autoencoder model with the given hyperparams.
    """
    switcher = {
        "vanilla": lambda: AE_vanilla(device, hyperparams),
        "classifier": lambda: AE_classifier(device, hyperparams),
        "variational": lambda: AE_variational(device, hyperparams),
        "sinkhorn": lambda: AE_sinkhorn(device, hyperparams),
        "sinkclass": lambda: AE_sinkclass(device, hyperparams),
    }
    model = switcher.get(ae_type, lambda: None)()
    if model is None:
        raise TypeError("Specified AE type does not exist!")

    return model


def define_torch_device() -> torch.device:
    # Use gpu for training if available. Alert the user if not and use cpu.
    print("\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        device = torch.device(
            "cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu"
        )
        if len(w):
            print(tcols.WARNING + "GPU not available." + tcols.ENDC)

    print("\033[92mUsing device:\033[0m", device)
    return device


def get_free_gpu(threshold_vram_usage=3000, max_gpus=1):
    """
    Returns the free gpu numbers on your system, to replace x in the string 'cuda:x'.
    The freeness is determined based on how much memory is currently being used on a
    gpu.

    Args:
        threshold_vram_usage: A GPU is considered free if the vram usage is below the
            threshold.
        max_gpus: Max GPUs is the maximum number of gpus to assign.
    """

    # Get the list of GPUs via nvidia-smi.
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )
    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    gpu_info = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]

    # Keep gpus under threshold only.
    free_gpus = [str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage]
    free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
    gpus_to_use = ",".join(free_gpus)

    if not gpus_to_use:
        raise RuntimeError(tcols.FAIL + "No free GPUs found." + tcols.ENDC)

    return gpus_to_use


def import_hyperparams(hyperparams_file) -> dict:
    """
    Import hyperparameters of an ae from json file.
    @model_path :: String of the path to a trained pytorch model folder
                   to import hyperparameters from the json file inside
                   that folder.

    returns :: Imported dictionary of hyperparams from .json file inside
        the trained model folder.
    """
    hyperparams_file = open(hyperparams_file)
    hyperparams = json.load(hyperparams_file)
    hyperparams_file.close()

    return hyperparams


def varname(index) -> str:
    """
    Gets the name of what variable is currently considered based on the
    index in the data array. Make sure the ordering is the same if data
    changes. Check the plots for consistency.
    @index :: Int of the variable number.

    returns :: The variable name.
    """
    jet_feat = [
        "$p_T$",
        "$\\eta$",
        "$\\phi$",
        "E",
        "$p_x$",
        "$p_y$",
        "$p_z$",
        "btag",
    ]
    num_jets = 7
    met_feat = ["$\\phi$", "$p_t$", "$p_x$", "$p_y$"]
    lep_feat = [
        "$p_t$",
        "$\\eta$",
        "$\\phi$",
        "Energy",
        "$p_x$",
        "$p_y$",
        "$p_z$",
    ]
    jet_nvar = len(jet_feat)
    met_nvar = len(met_feat)
    lep_nvar = len(lep_feat)

    if index < jet_nvar * num_jets:
        jet = index // jet_nvar + 1
        var = index % jet_nvar
        varstring = "Jet " + str(jet) + " " + jet_feat[var]
        return varstring
    index -= jet_nvar * num_jets

    if index < met_nvar:
        var = index % met_nvar
        varstring = "MET " + met_feat[var]
        return varstring
    index -= met_nvar

    if index < lep_nvar:
        var = index % lep_nvar
        varstring = "Lepton " + lep_feat[var]
        return varstring

        return None
