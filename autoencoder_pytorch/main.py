# Autoencoder that reduces the number of features from 67 to 8.
import os, sys, torch, torchvision, time, argparse, warnings
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import tensorData, AE
from train import train

start_time = time.time()
seed = 100
torch.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)

# Use gpu if available.
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:',device)

defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--training_file", type=str, default=infiles,
    help="The path to the training data.")
parser.add_argument("--validation_file", type=str, default=infiles,
    help="The path to the validation data.")
parser.add_argument('--lr',type=float,default=2e-03,
    help='The learning rate.')
parser.add_argument('--layers', type=int, default=defaultlayers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, default=64,
    help='The batch size.')
parser.add_argument('--epochs', type=int, default=85,
    help='The number of training epochs.')
parser.add_argument('--fileFlag', type=str, default='',
    help='fileFlag to concatenate to filetag')
args = parser.parse_args()

def load_train_valid(train, valid):
    """
    Load the training and validation data into pytorch dataloader objects.

    @train :: The training data set.
    @valid :: The validation data set.

    @returns ::
    """
    feature_size = training_data.shape[1]
    # Insert the input dimensions at the beginning of the list.
    layers.insert(0, feature_size)

    validation_size = validation_data.shape[0]

    # Convert to torch dataset:
    dataset = tensorData(dataset)
    validDataset = tensorData(validDataset)

    train_loader = torch.utils.data.DataLoader(train,
        batch_size=args.batch, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid,
        batch_size=validation_size, shuffle = True)

    return train_loader, valid_loader

def prepare_outdir(model_nodes):
    # Prepare to the naming of training outputs. Do not print output size.
    layersTag = '.'.join(str(inode) for inode in model_nodes[1:])
    filetag = 'L' + layersTag + '_B' + str(args.batch) + '_Lr{:.0e}'.\
        format(args.lr) + "_" + fileFlag

    outdir = './trained_models/' + filetag + '/'
    if not(os.path.exists(outdir)): os.mkdir(outdir)

    return filetag, outdir

def diagnosis_plots(loss_train, loss_valid, min_valid, nodes):
    # Quick plots to see if what we trained is of any good quickly.
    plt.plot(list(range(args.epochs)), loss_train,
        label='Training Loss (last batch)')
    plt.plot(list(range(args.epochs)), loss_valid,
        label='Validation Loss (1 per epoch)')
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    plt.title("B = " + str(args.batch) + ", lr=" + str(args.lr) + ", " +
        str(nodes) + ', L ={:.6f}'.format(min_valid))
    plt.legend()
    plt.savefig(outdir + 'loss_epochs.png')

def save_MSE_log(filetag, train_time, min_valid):
    # Save MSE log to check best model:
    with open('trained_models/mse_log.txt','a+') as mse_log:
       log_entry = filetag + f': Training time = {train_time:.2f} min, Min. Validation loss = {min_valid:.6f}\n'
    mseLog.write(logEntry)

if __name__ == '__main__':

    # Sig + Bkg training, validation, and testing import.
    training_data   = np.load(args.training_file)
    validation_data = np.load(args.validation_file)

    # Put the data into pytorch data loader objects.
    train_loader, valid_loader=load_train_valid(training_data, validation_data)
    model = AE(node_number = args.lr).to(device)

    # Create an optimizer object.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Mean-squared error loss.
    criterion = nn.MSELoss(reduction = 'mean')

    print('\n---\nBatch size = ' + str(args.batch) + '\n Learning rate = ' +
        str(args.lr) + '\nLayers = ' + str(model.node_number))

    filetag, outdir = prepare_outdir(model.node_number)
    # Print model architecture in output file:
    with open(outdir + 'model_architecture.txt', 'w') as model_architecture:
	   print(model, file=model_architecture)

    # Train the autoencoder!
    loss_training, loss_validation, minimum_validation =
        train(train_loader,valid_loader, model, criterion, optimizer,
            args.epochs, device, outdir)

    diagnosis_plots(loss_train, loss_valid, min_valid, model.node_number)

    # Time it and save the MSE log to see what the best model is.
    end_time = time.time()
    train_time = (end_time - start_time)/60

    save_MSE_log(filetag, train_time, min_valid)
