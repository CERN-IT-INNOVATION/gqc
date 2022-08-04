# Runs the neural network training. The normalized or standardized data is imported,
# and the classifier model is defined, given the specified options/hyperparms.
# The model is then trained and a loss plot is saved, along with the
# architecture of the network, its hyperparameters, and the best model weights.

import time
import os
import argparse
import sys
sys.path.append("..")
import torch
torch.manual_seed(0)

from autoencoders import util as ae_util
from autoencoders import data
from vqc_pennylane.terminal_colors import tcols
from vqc_pennylane import util
from vqc_pennylane import qdata as qd
from neural_network import NeuralNetwork

def main():
    args = get_arguments()
    device = ae_util.define_torch_device()
    outdir = "./trained_nns/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Data loading. FIXME: this or through qdata to ensure identical loading 
    # with the hybrid VQC?
    #ae_data = data.AE_data(
    #    args["data_folder"],
    #    args["norm"],
    #    args["nevents"],
    #    args["ntrain"],
    #    args["nvalid"],
    #    seed=args["seed"],
    #)
    qdata = qd.qdata(
    args["data_folder"],
    args["norm"],
    args["nevents"],
    args["model_path"],
    train_events=args["ntrain"],
    valid_events=args["nvalid"],
    seed=args["seed"],
    )
    train_loader, valid_loader, _ = util.get_hybrid_data(qdata, args)
    #train_loader = ae_data.get_loader("train", device, args["batch_size"], True)
    #valid_loader = ae_data.get_loader("valid", device, None, True)

    model = NeuralNetwork(device, args)

    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    time_the_training(
        model.train_model, train_loader, valid_loader, args["epochs"], 20, outdir
    )

    model.loss_plot(outdir)

def get_arguments() -> dict:
    """
    Parses command line arguments and gives back a dictionary.
    
    Returns: Dictionary with the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_folder", type=str,
                        help="The folder where the data is stored on the system..")
    parser.add_argument("--norm", type=str,
                        help="The name of the normalisation that you'll to use.")
    parser.add_argument('--model_path', type=str,
                        help="The path to the Auto-Encoder model.")
    parser.add_argument("--nevents", type=str,
                        help="The number of signal events of the norm file.")
    parser.add_argument("--ntrain", type=int, default=-1,
                        help="The exact number of training events used < nevents.")
    parser.add_argument("--nvalid", type=int, default=-1,
                        help="The exact number of valid events used < nevents.")
    parser.add_argument('--lr', type=float, default=2e-03,
                        help="The learning rate.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="The batch size.")
    parser.add_argument("--epochs", type=int, default=85,
                        help="The number of training epochs.")
    parser.add_argument("--outdir", type=str, default="",
                        help="Flag the file in a certain way for easier labeling.")
    args = parser.parse_args()
    
    seed = 12345

    args = {
        "data_folder": args.data_folder,
        "norm": args.norm,
        "nevents":      args.nevents,
        "model_path": args.model_path,
        "ntrain": args.ntrain,
        "nvalid": args.nvalid,
        "ae_layers":    [67, 64, 52, 44, 32, 24, 16, 1],
        "batch_size": args.batch_size,
        "lr":           args.lr,
        "epochs":       args.epochs,
        "out_activ":    'nn.Sigmoid()',
        "adam_betas":   (0.9, 0.999),
        "outdir":       args.outdir,
        "seed": seed,
    }
    return args

def time_the_training(train: callable, *args):
    """Times the training of the neural network.

    Args:
        train (callable): The training method of the NeuralNetwork class.
        *args: Arguments for the train_model method.
    """
    train_time_start = time.perf_counter()
    train(*args)
    train_time_end = time.perf_counter()
    print(
        tcols.OKCYAN +
        f"Training completed in: {train_time_end-train_time_start:.2e} s or "
        f"{(train_time_end-train_time_start)/3600:.2e} h." + tcols.ENDC
    )

if __name__ == "__main__":
    main()