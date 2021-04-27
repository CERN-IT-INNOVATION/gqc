# Main script to run the autoencoders. The point is to reduce the number of
# features from 67 to 8 or generally to a lower nubmer manageable for quantum
# classifiers to perform on near-term quantum computers.

import time, argparse
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


default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--training_file", type=str,
    help="The path to the training data.")
parser.add_argument("--validation_file", type=str,
    help="The path to the validation data.")
parser.add_argument('--lr',type=float,default=2e-03,
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
    # Only parse args if ran as main script.
    args = parser.parse_args()
    # Define torch device
    device = util.define_torch_device()

    # Load the data.
    train_data   = np.load(args.training_file)
    valid_data   = np.load(args.validation_file)
    train_loader = util.to_pytorch_data(train_data, args.batch, True)
    valid_loader = util.to_pytorch_data(valid_data, args.batch, True)

    # Insert the input dimensions at the beginning of the layer list.
    (args.layers).insert(0, train_data.shape[1])

    # Define model, optimizer, and mean squared loss criterion.
    model = model_vasilis.AE(node_number=args.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')

    print('---\nBatch size = ' + str(args.batch) + '\nLearning rate = ' +
        str(args.lr) + '\nLayers = ' + str(model.node_number))

    # Print out model architecture.
    filetag, outdir = util.prepare_output(model.node_number, args.batch,
        args.lr, args.file_flag)
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
       print(model, file=model_architecture)

    # Start timer.
    start_time = time.time()

    loss_train, loss_valid, min_valid = model_vasilis.train(train_loader,
        valid_loader, model, criterion, optimizer, args.epochs, device, outdir)

    # Stop timer.
    end_time = time.time()
    train_time = (end_time - start_time)/60

    plotting.diagnosis_plots(loss_train, loss_valid, min_valid,
        model.node_number, args.batch, args.lr, args.epochs, outdir)
    util.save_MSE_log(filetag, train_time, min_valid)
