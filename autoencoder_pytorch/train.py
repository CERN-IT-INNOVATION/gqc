# Runs the autoencoder(s). The normalized or standardized data is imported,
# the atuencoder model is defined imported, given a number of layers,
# a learning rate, a device (gpu or cpu), and encoder and decoder activation
# functions (legacy version had sigmoids). The model is then trained and
# a loss plot is saved, along with the model that showed the lowest loss
# during training and validation and its architecture.

import time, argparse
import numpy as np
import os

import torch.nn as nn

import util
import data
from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
parser.add_argument("--aetype", type=str,
    help="The type of autoencoder that you will use, i.e., vanilla etc..")
parser.add_argument('--lr', type=float, default=2e-03,
    help='The learning rate.')
parser.add_argument('--batch', type=int, default=128,
    help='The batch size.')
parser.add_argument('--epochs', type=int, default=85,
    help='The number of training epochs.')
parser.add_argument('--outdir', type=str, default='',
    help='Flag the file in a certain way for easier labeling.')

def main():
    args   = parser.parse_args()
    device = util.define_torch_device()
    vqc_specs   = [["zzfm", 0, 4], ["2local", 0, 20, 4, "linear"],
                   ["zzfm", 4, 8], ["2local", 20, 40, 4, "linear"]]
    hyperparams   = {
        "lr"           : args.lr,
        "ae_layers"    : [64, 52, 44, 32, 24, 16],
        "class_layers" : [128, 64, 32, 16, 8, 1],
        "enc_activ"    : 'nn.Tanh()',
        "dec_activ"    : 'nn.Tanh()',
        "vqc_specs"    : vqc_specs,
        "loss_weight"  : 1,
        "weight_sink"  : 1,
        "adam_betas"   : (0.9, 0.999),
    }
    outdir = "./trained_models/" + args.outdir + '/'
    if not os.path.exists(outdir): os.makedirs(outdir)

    # Load the data.
    ae_data = data.AE_data(args.data_folder, args.norm, args.nevents)
    train_loader = ae_data.get_loader("train", device, args.batch, True)
    valid_loader = ae_data.get_loader("valid", device, None, True)

    # Define the model and prepare the output folder.
    (hyperparams['ae_layers']).insert(0, ae_data.nfeats)
    model = util.choose_ae_model(args.aetype, device, hyperparams)

    # Train and time it.
    start_time = time.time()
    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    model.train_autoencoder(train_loader, valid_loader, args.epochs, outdir)
    end_time = time.time()

    train_time = (end_time - start_time)/60
    print(f"Training time: {train_time:.2e} mins.")

    model.loss_plot(outdir)

if __name__ == '__main__':
    main()
