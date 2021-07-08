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

import vanilla_ae
import classifier_ae
import plotting
import util

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--train_file", type=str,
    help="The name of the training data file.")
parser.add_argument("--valid_file", type=str,
    help="The name of the validation data file.")
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
    args = parser.parse_args()
    device = util.define_torch_device()

    # Load the data.
    train_data = np.load(os.path.join(args.data_folder, args.train_file))
    valid_data = np.load(os.path.join(args.data_folder, args.valid_file))
    train_target_file = "y" + args.train_file[1:]
    valid_target_file = "y" + args.valid_file[1:]
    train_target = np.load(os.path.join(args.data_folder, train_target_file))
    valid_target = np.load(os.path.join(args.data_folder, valid_target_file))
    train_loader = util.to_pytorch_data(train_data, device, args.batch, True)
    valid_loader = util.to_pytorch_data(valid_data, device, None, True)
    print("\n----------------")
    print("\033[92mData loading complete:\033[0m")
    print(f"Training data size: {train_data.shape[0]:.2e}")
    print(f"Validation data size: {valid_data.shape[0]:.2e}")
    print("----------------\n")

    # Define the model and prepare the output folder.
    (args.layers).insert(0, len(train_loader.dataset[1]))
    model = choose_ae_model("vanilla", device, args.layers, args.lr,
        nn.Sigmoid(), nn.Sigmoid(), None, None, None)
    outdir = util.prepare_output(model, args.batch, args.lr,
        len(train_loader.dataset), args.file_flag)

    # Train and time it.
    start_time = time.time()

    loss_train, loss_valid, min_valid = \
        model.train_model(train_loader, valid_loader, args.epochs, outdir)

    end_time = time.time()
    train_time = (end_time - start_time)/60
    print(f"Training time: {train_time:.2e} mins.")

    plotting.loss_plot(loss_train, loss_valid, min_valid, model.nodes,
        args.batch, model.lr, args.epochs, outdir)

def choose_ae_model(user_choice, device, layers, lr, en_activ=nn.Tanh(),
    dec_activ=nn.Tanh(), class_layers=[256, 256, 128, 64, 32, 1],
    recon_weight=0.5, class_weight=0.5):

    switcher = {
        "vanilla":   lambda : vanilla_ae_model(device, layers, lr, en_activ,
            dec_activ),
        "classifer": lambda : classifier_ae_model(device, layers, lr, en_activ,
            dec_activ, class_layers, recon_weight, class_weight)
    }

    func   = switcher.get(user_choice, lambda : "Invalid type of AE given!")
    model = func()

    return model

def vanilla_ae_model(device, layers, lr, en_activ, dec_activ):

    return vanilla_ae.AE(nodes=layers, lr=lr, device=device, en_activ=en_activ,
        dec_activ=dec_activ).to(device)

def classifier_ae_model(device, layers, lr, en_activ, dec_activ, class_layers,
    recon_weight, class_weight):

    return classifier_ae.AE(nodes=layers,lr=lr,device=device,en_activ=en_activ,
        dec_activ=dec_activ).to(device)

if __name__ == '__main__':
    main()
