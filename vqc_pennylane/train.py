# Main script of the vqc. Imports the data and runs the training of the
# The VQC training script. Here, the vqc class is instantiated with some
# parameters, the circuit is built, and then it is trained on a data set.
# The hyperparameters of the circuit, the best weights, and a plot of the
# loss function evolution throughout the epochs are saved in a folder.
import os
from time import perf_counter
from typing import Tuple

from .vqc import VQC
from .terminal_colors import tcols
from . import qdata as qd
from . import util
from .vqc_hybrid import VQCHybrid



def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["ae_model_path"],
        train_events=args["train_events"],
        valid_events=args["valid_events"],
        seed=args["seed"]
    )
    outdir = "./trained_vqcs/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model, train_loader, valid_loader = get_data_and_model(qdata, args)

    model.export_hyperparameters(outdir)

    train_time_start = perf_counter()
    model.train_model(train_loader, valid_loader, args["epochs"], 20, outdir)
    train_time_end = perf_counter()
    print(f"Training completed in: {train_time_end-train_time_start:.2e} s")

    model.loss_plot(outdir)


def get_data_and_model(qdata_loader, args) -> Tuple:
    """
    Method that choses the training type.
    """
    if args["hybrid_training"]:
        vqc_hybrid = VQCHybrid(qdevice='default.qubit', device='cpu', hpars=args)
        return vqc_hybrid, *get_hybrid_training_data(qdata_loader, args)
    else: 
        vqc = VQC("default.qubit", args)
        util.print_model_info(args["ae_model_path"], qdata_loader, vqc)
        return vqc, *get_nonhybrid_training_data(qdata_loader, args)

def get_nonhybrid_training_data(qdata_loader, args) -> Tuple:
    """
    Loads the data from pre-trained Autoencoder latent space when we have non 
    hybrid VQC training.
    """
    train_features = qdata_loader.batchify(qdata_loader.get_latent_space("train"),
                                    args["batch_size"])
    train_labels = qdata_loader.batchify(qdata_loader.ae_data.trtarget, args["batch_size"])
    train_loader = [train_features, train_labels]

    valid_features = qdata_loader.get_latent_space("valid")
    valid_labels = qdata_loader.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]
    return train_loader, valid_loader

def get_hybrid_training_data(qdata_loader, args) -> Tuple:
    """
    Loading the raw input data for hybrid training.
    """
    train_loader = qdata_loader.ae_data.get_loader("train", "cpu", args["batch_size"], True)
    valid_loader = qdata_loader.ae_data.get_loader("valid", "cpu", shuffle=True)
    return train_loader, valid_loader

