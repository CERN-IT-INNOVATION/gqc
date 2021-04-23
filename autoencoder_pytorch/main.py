# Autoencoder that reduces the number of features from 67 to 8.
import os, sys, time, argparse, warnings
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim

import model_vasilis
import plotting
import util

seed = 100
torch.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)

# Use gpu if available.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:',device)
default_layers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--training_file", type=str, default=infiles,
    help="The path to the training data.")
parser.add_argument("--validation_file", type=str, default=infiles,
    help="The path to the validation data.")
parser.add_argument('--lr',type=float,default=2e-03,
    help='The learning rate.')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, default=64,
    help='The batch size.')
parser.add_argument('--epochs', type=int, default=85,
    help='The number of training epochs.')
parser.add_argument('--fileFlag', type=str, default='',
    help='fileFlag to concatenate to filetag')
args = parser.parse_args()

if __name__ == '__main__':

    # Load the data.
    train_data = np.load(args.training_file)
    valid_data = np.load(args.validation_file)
    train_loader = util.to_pytorch_data(train_data, args.batch, True)
    valid_loader = util.to_pytorch_data(valid_data, args.batch, True)

    # Insert the input dimensions at the beginning of the layer list.
    (args.layers).insert(0, training_data.shape[1])

    # Define model, optimizer, and mean squared loss criterion.
    model = model_vasilis.AE(node_number=args.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')

    print('\n---\nBatch size = ' + str(args.batch) + '\n Learning rate = ' +
        str(args.lr) + '\nLayers = ' + str(model.node_number))

    # Print out model architecture.
    filetag, outdir = util.prepare_output(model.node_number)
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
       print(model, file=model_architecture)

    # Start timer.
    start_time = time.time()

    loss_training, loss_validation, minimum_validation = model_vasilis.train(
        train_loader,valid_loader, model, criterion, optimizer, args.epochs,
        device, outdir)

    # Stop timer.
    end_time = time.time()
    train_time = (end_time - start_time)/60

    plotting.diagnosis_plots(loss_train, loss_valid, min_valid,
        model.node_number)
    util.save_MSE_log(filetag, train_time, min_valid)
