# Main script to run the autoencoders. The point is to reduce the number of
# features from 67 to 8 or generally to a lower nubmer manageable for quantum
# classifiers to perform on near-term quantum computers.

import time, argparse
import numpy as np

import model_vasilis as basic_nn
import model_vasilis_tanh as tanh_nn
import model_vasilis_improved_loss as new_loss_nn
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
    train_loader, valid_loader = \
    util.get_train_data(args.train_file, args.valid_file, args.batch, device)

    # Define the model.
    (args.layers).insert(0, len(train_loader.dataset[1]))
    model = relu_nn.AE(nodes=args.layers,lr=args.lr,device=device).to(device)

    # Print out model architecture.
    filetag, outdir = util.prepare_output(model.nodes, args.batch, args.lr,
        len(train_loader.dataset), args.file_flag)
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
       print(model, file=model_architecture)

    # Train the model.
    start_time = time.time()
    loss_train, loss_valid, min_valid = relu_nn.train(
        train_loader, valid_loader, model, args.epochs, outdir)
    end_time = time.time()
    train_time = (end_time - start_time)/60

    print("Training time: {:.2e} mins.".format(train_time))

    plotting.diagnosis_plots(loss_train, loss_valid, min_valid, model.nodes,
        args.batch, args.lr, args.epochs, outdir)
    util.save_MSE_log(filetag, train_time, min_valid, outdir)
