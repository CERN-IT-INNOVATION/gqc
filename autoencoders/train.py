# Runs the autoencoder. The normalized or standardized data is imported,
# and the autoencoder model is defined, given the specified options.
# The model is then trained and a loss plot is saved, along with the
# architecture of the model, its hyperparameters, and the best model weights.

import time
import os

from . import util
from . import data
from .terminal_colors import tcols


def main(args):
    device = util.define_torch_device()
    outdir = "./trained_aes/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Data loading.
    ae_data = data.AE_data(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["train_events"],
        args["valid_events"],
    )
    train_loader = ae_data.get_loader("train", device, args["batch"], True)
    valid_loader = ae_data.get_loader("valid", device, None, True)

    # AE model definition.
    (args["ae_layers"]).insert(0, ae_data.nfeats)
    model = util.choose_ae_model(args["aetype"], device, args)

    start_time = time.time()
    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    model.train_autoencoder(train_loader, valid_loader, args["epochs"], outdir)
    end_time = time.time()

    train_time = (end_time - start_time) / 60
    print(tcols.OKCYAN + f"Training time: {train_time:.2e} mins." + tcols.ENDC)

    model.loss_plot(outdir)
