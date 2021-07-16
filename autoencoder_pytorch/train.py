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

import plot
import util

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
parser.add_argument('--lr', type=float, default=2e-03,
    help='The learning rate.')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layer structure.')
parser.add_argument('--batch', type=int, default=128,
    help='The batch size.')
parser.add_argument('--epochs', type=int, default=85,
    help='The number of training epochs.')
parser.add_argument('--file_flag', type=str, default='',
    help='Flag the file in a certain way for easier labeling.')

def main():
    args               = parser.parse_args()
    device             = util.define_torch_device()
    ae_type            = "classifier"
    encoder_activation = nn.Tanh()
    decoder_activation = nn.Tanh()

    # Get the names of the data files. We follow a naming scheme. See util mod.
    train_file = util.get_train_file(args.norm, args.nevents)
    valid_file = util.get_valid_file(args.norm, args.nevents)
    train_target_file = util.get_train_target(args.norm, args.nevents)
    valid_target_file = util.get_valid_target(args.norm, args.nevents)

    # Load the data.
    train_data   = np.load(os.path.join(args.data_folder, train_file))
    valid_data   = np.load(os.path.join(args.data_folder, valid_file))
    train_target = np.load(os.path.join(args.data_folder, train_target_file))
    valid_target = np.load(os.path.join(args.data_folder, valid_target_file))

    train_loader = \
        util.to_pytorch_data(train_data, train_target, device, args.batch, True)
    valid_loader = \
        util.to_pytorch_data(valid_data, valid_target, device, None, True)

    print("\n----------------")
    print("\033[92mData loading complete:\033[0m")
    print(f"Training data size: {train_data.shape[0]:.2e}")
    print(f"Validation data size: {valid_data.shape[0]:.2e}")
    print("----------------\n")

    # Define the model and prepare the output folder.
    nfeatures = train_data.shape[1]
    (args.layers).insert(0, nfeatures)

    model = util.choose_ae_model(ae_type, device, args.layers, args.lr,
        encoder_activation, decoder_activation, loss_weight=1,
        class_layers=[64,64,32,16,8,1])
    outdir = util.prep_out(model, args.batch, args.lr, args.nevents,
        args.norm, args.file_flag)

    # Train and time it.
    start_time = time.time()

    loss_train, loss_valid, min_valid = \
        model.train_autoencoder(train_loader, valid_loader, args.epochs, outdir)

    end_time = time.time()
    train_time = (end_time - start_time)/60
    print(f"Training time: {train_time:.2e} mins.")

    plot.loss_plot(loss_train, loss_valid, min_valid, args.epochs, outdir)

if __name__ == '__main__':
    main()
