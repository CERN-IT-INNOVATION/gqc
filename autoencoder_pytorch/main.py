# Runs the autoencoder(s). The normalized or standardized data is imported,
# the atuencoder model is defined imported, given a number of layers,
# a learning rate, a device (gpu or cpu), and encoder and decoder activation
# functions (legacy version had sigmoids). The model is then trained and
# a loss plot is saved, along with the model that showed the lowest loss
# during training and validation and its architecture.

import time, argparse
import numpy as np

import torch.nn as nn

import vanilla_ae
import plotting
import util

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_file", type=str,
    help="The path to the training data.")
parser.add_argument("--valid_file", type=str,
    help="The path to the validation data.")
parser.add_argument('--lr', type=float, default=2e-03,
    help='The learning rate.')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, default=64,
    help='The batch size.')
parser.add_argument('--epochs', type=int, default=85,
    help='The number of training epochs.')
parser.add_argument('--file_flag', type=str, default='',
    help='Flag the file in a certain way for easier labeling.')

if __name__ == '__main__':
    args = parser.parse_args()
    device = util.define_torch_device()

    # Load the data.
    train_data = np.load(args.train_file)
    valid_data = np.load(args.valid_file)
    print("\n----------------")
    print(f"Training data size: {train_data.shape[0]:.2e}")
    print(f"Validation data size: {valid_data.shape[0]:.2e}")
    print("----------------\n")
    train_loader = util.to_pytorch_data(train_data, device, args.batch, True)
    valid_loader = util.to_pytorch_data(valid_data, device, None, True)
    (args.layers).insert(0, len(train_loader.dataset[1]))

    # Define the model and prepare the output folder.
    model = vanilla_ae.AE(nodes=args.layers, lr=args.lr,
        device=device, en_activ=nn.Tanh(), dec_activ=nn.Tanh()).to(device)
    outdir = util.prepare_output(model, args.batch, args.lr,
        len(train_loader.dataset), args.file_flag)

    # Train and time it.
    start_time = time.time()

    loss_train, loss_valid, min_valid = \
        vanilla_ae.train(train_loader, valid_loader, model, args.epochs, outdir)

    end_time = time.time()
    train_time = (end_time - start_time)/60
    print(f"Training time: {train_time:.2e} mins.")

    plotting.loss_plot(loss_train, loss_valid, min_valid, model.nodes,
        args.batch, args.lr, args.epochs, outdir)
