# Main script of the vqc. Imports the data and runs the training of the
# The VQC training script. Here, the vqc class is instantiated with some
# parameters, the circuit is built, and then it is trained on a data set.
# The hyperparameters of the circuit, the best weights, and a plot of the
# loss function evolution throughout the epochs are saved in a folder.
import os
from time import perf_counter
import numpy as np

from .vqc import VQC
from .terminal_colors import tcols
from . import qdata as qd
from . import util


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["ae_model_path"],
        train_events=args["train_events"],
        valid_events=args["valid_events"],
        seed=123
    )
    outdir = "./trained_vqcs/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_features = qdata.batchify(qdata.get_latent_space("train"),
                                    args["batch_size"])
    train_labels = qdata.batchify(qdata.ae_data.trtarget, args["batch_size"])
    train_loader = [train_features, train_labels]

    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]

    vqc = VQC("default.qubit", args)
    vqc.export_hyperparameters(outdir)
    util.print_model_info(args["ae_model_path"], qdata, vqc)

    train_time_start = perf_counter()
    vqc.train_vqc(train_loader, valid_loader, args["epochs"], 20, outdir)
    train_time_end = perf_counter()
    print(f"Training completed in: {train_time_end-train_time_start:.2e} s")

    vqc.loss_plot(outdir)
