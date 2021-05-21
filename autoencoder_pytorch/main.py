# Main script to run the autoencoders. The point is to reduce the number of
# features from 67 to 8 or generally to a lower nubmer manageable for quantum
# classifiers to perform on near-term quantum computers.

import time, argparse
import numpy as np

import model_vasilis
import model_vasilis_tanh
import plotting
import util

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--training_file", type=str,
    help="The path to the training data.")
parser.add_argument("--validation_file", type=str,
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
    train_loader, valid_loader = util.get_train_data(args.training_file,
        args.validation_file, args.batch, device)

    # Define the model.
    (args.layers).insert(0, len(train_loader.dataset[1]))
    model = model_vasilis_improved.AE(node_number=args.layers, lr=args.lr).to(device)

    # Print out model architecture.
    filetag, outdir = util.prepare_output(model.node_number, args.batch,
        args.lr, len(train_loader.dataset), args.file_flag)
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
       print(model, file=model_architecture)

    # Train the model.
    start_time = time.time()
    loss_train, loss_valid, min_valid = model_vasilis_improved.train(
        train_loader, valid_loader, model, device, args.epochs, outdir)
    end_time = time.time()

    train_time = (end_time - start_time)/60
    print("Training time: {:.2e} mins.".format(train_time), flush=True)

    plotting.diagnosis_plots(loss_train, loss_valid, min_valid,
        model.node_number, args.batch, args.lr, args.epochs, outdir)
    util.save_MSE_log(filetag, train_time, min_valid, outdir)
