# Plot the different things with respect to a trained autoencoder model
# such as the epochs vs loss plot.
import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import torch
import torch.nn as nn
from sklearn import metrics

import ae_vanilla
import util
from util import compute_model

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--valid_file", type=str,
    help="The name of the validation file.")
parser.add_argument("--test_file", type=str,
    help="The name of the test file.")
parser.add_argument("--test_target", type=str,
    help="The name of the test target.")
parser.add_argument('--model_path', type=str, required=True,
    help="The path to the saved model.")

def main():
    args = parser.parse_args()
    device = 'cpu'

    # Import the hyperparameters used in training and the data.
    layers, batch, lr = import_hyperparams(args.model_path)
    valid_data   = np.load(os.path.join(args.data_folder, args.valid_file))
    test_data    = np.load(os.path.join(args.data_folder, args.test_file))
    valid_loader = util.to_pytorch_data(valid_data, device, None, False)
    test_loader  = util.to_pytorch_data(test_data, device, None, False)

    test_data   = np.load(os.path.join(args.data_folder, args.test_file))
    test_target = np.load(os.path.join(args.data_folder, args.test_target))
    data_sig, data_bkg = util.split_sig_bkg(test_data, test_target)

    test_loader_sig = util.to_pytorch_data(data_sig, device, None, False)
    test_loader_bkg = util.to_pytorch_data(data_bkg, device, None, False)

    # Import the model.
    (layers).insert(0, test_data.shape[1])
    model = util.load_model(ae_vanilla.AE, layers, lr, args.model_path, device,
        en_activ=nn.Tanh(), dec_activ=nn.Tanh())
    criterion = model.criterion()

    # Compute criterion scores for data.
    compute_losses(model, criterion, valid_loader, test_loader)

    # Compute the signal and background latent spaces and decoded data.
    output_sig, sig_latent, input_sig = compute_model(model, test_loader_sig)
    output_bkg, bkg_latent, input_bkg = compute_model(model, test_loader_bkg)
    sig_latent = sig_latent.numpy()
    bkg_latent = bkg_latent.numpy()

    latent_sig_bkg = np.vstack((sig_latent, bkg_latent))
    target_sig_bkg = \
    np.concatenate((np.ones(sig_latent.shape[0]),np.zeros(bkg_latent.shape[0])))

    # Do the plots.
    latent_space_plots(sig_latent, bkg_latent, args.model_path)
    latent_roc_plots(latent_sig_bkg, target_sig_bkg, args.model_path)
    sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg,args.model_path)


def import_hyperparams(model_path):
    """
    Imports the hyperparameters of a given model from the model path given by
    the user of the autoencoder that was trained.

    @model_path :: String of path to the autoencoder folder.

    @returns :: Hyperparameters of the autoencoder.
    """
    hyperparams = max(model_path.split('/'), key=len)
    layers  = hyperparams[hyperparams.find('L')+1:hyperparams.find('_')]
    layers  = [int(nb) for nb in layers.split(".")]
    batch   = int(hyperparams[hyperparams.find('B')+1:hyperparams.find('_',
        hyperparams.find('B')+1, len(hyperparams))])
    lr      = float(hyperparams[hyperparams.find('Lr')+2:hyperparams.find('_',
        hyperparams.find('Lr')+2, len(hyperparams))])

    print("\nImported model hyperparameters:")
    print("--------------------------------")
    print(f"Layers: {layers}")
    print(f"Batch: {batch}")
    print(f"Learning Rate: {lr}")

    return layers, batch, lr

def compute_losses(model, criterion, valid_data, test_data):
    # Computes the scores of the model on the test and train datasets and
    # saves the results to a file.
    test_output, _, test_input = compute_model(model, test_data)
    valid_output, _, valid_input = compute_model(model, valid_data)

    print('\n----------------------------------')
    print(f"TEST MSE: {criterion(test_output, test_input).item()}")
    print(f"VALID MSE: {criterion(valid_output, valid_input).item()}")
    print('----------------------------------\n')

def ratio_plotter(input_data, output_data, ifeature, color, class_label=''):
    # Plots two overlaid histograms.
    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    prange = (np.amin(input_data, axis=0), np.amax(input_data, axis=0))
    plt.hist(x=input_data, bins=60, range=prange, alpha=0.8, histtype='step',
        linewidth=2.5, label=class_label, density=True, color=color)
    plt.hist(x=output_data, bins=60, range=prange, alpha=0.8, histtype='step',
        linewidth=2.5, label='Rec. ' + class_label, linestyle='dashed',
        density=True, color=color)

    plt.xlabel(util.varname(ifeature) + ' (normalized)'); plt.ylabel('Density')
    plt.xlim(*prange)
    plt.gca().set_yscale("log")
    plt.legend()

def latent_space_plots(latent_data_sig, latent_data_bkg, model_path):
    # Makes the plots of the latent space data produced by the encoder.
    storage_folder_path = model_path + 'latent_plots/'
    if not os.path.exists(storage_folder_path): os.makedirs(storage_folder_path)

    for i in range(latent_data_sig.shape[1]):
        xmax = max(np.amax(latent_data_sig[:,i]),np.amax(latent_data_bkg[:,i]))
        xmin = min(np.amin(latent_data_sig[:,i]),np.amin(latent_data_bkg[:,i]))

        hSig,_,_ = plt.hist(x=latent_data_sig[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.8, histtype='step',
            linewidth=2.5, label='Sig', color='navy')
        hBkg,_,_ = plt.hist(x=latent_data_bkg[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.4, histtype='step',
            linewidth=2.5,label='Bkg', color='gray', hatch='xxx')

        plt.legend()
        plt.xlabel(f'Latent feature {i}')
        plt.savefig(storage_folder_path + 'Latent Feature '+ str(i) + '.pdf')
        plt.close()

    print(f"\033[92mLatent plots were saved to {storage_folder_path}.\033[0m")

def latent_roc_plots(data, target, model_path):
    """
    Compute the roc curve of a given 2D dataset of features.

    @data   :: 2D array, each column is a feature and each row an event.
    @target :: 1D array, each element is 0 or 1 corresponding to bkg or sig.

    @returns :: Saves the the roc curves of all the feats along with an
    indication of the AUC on top of it.
    """
    roc_auc_plot_dir = model_path + 'latent_roc_plots/'
    if not os.path.exists(roc_auc_plot_dir): os.makedirs(roc_auc_plot_dir)

    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    auc_sum = 0.
    for feature in range(data.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(target, data[:, feature])
        auc = metrics.roc_auc_score(target, data[:, feature])

        fig = plt.figure(figsize=(12, 10))
        plt.title(f"Latent feature {feature}")
        plt.plot(fpr, tpr, label=f"AUC: {auc}", color='navy')
        plt.plot([0, 1], [0, 1], ls="--", color='gray')

        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
        plt.legend()

        auc_sum += auc
        fig.savefig(roc_auc_plot_dir + f"Latent feature {feature}.png")
        plt.close()

    with open(roc_auc_plot_dir + 'auc_sum.txt', 'w') as auc_sum_file:
        auc_sum_file.write(f"{auc_sum:.3f}")

    print(f"\033[92mLatent roc plots were saved to {roc_auc_plot_dir}.\033[0m")

def sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg, model_path):
    # Plot the background and the signal distributions for the input data and
    # the reconstructed data, overlaid.
    storage_folder_path = model_path + 'ratio_plots/'
    if not os.path.exists(storage_folder_path): os.makedirs(storage_folder_path)

    for idx in range(input_sig.numpy().shape[1]):
        plt.figure(figsize=(12,10))

        ratio_plotter(input_bkg.numpy()[:,idx], output_bkg.numpy()[:,idx], idx,
            'gray', class_label='Background')
        ratio_plotter(input_sig.numpy()[:,idx], output_sig.numpy()[:,idx], idx,
            'navy', class_label='Signal')

        plt.savefig(storage_folder_path + 'Ratio Plot ' +
            util.varname(idx) + '.pdf')
        plt.close()

    print(f"\033[92mRatio plots were saved to {storage_folder_path}.\033[0m")

def loss_plot(loss_train, loss_valid, min_valid, nodes, batch_size, lr,
    epochs, outdir):
    # Plot the epochs vs loss for the loss of the last batch of the training
    # and also the loss computed on the validation data. No test data is used.
    plt.plot(list(range(epochs)), loss_train, color='gray',
        label='Training Loss (last batch)')
    plt.plot(list(range(epochs)), loss_valid, color='navy',
        label='Validation Loss (1 per epoch)')
    plt.xlabel("epochs"); plt.ylabel("MSE")
    plt.title("B=" + str(batch_size) + ", lr=" + str(lr) + ", L=" +
        str(nodes) + ', min={:.6f}'.format(min_valid))

    plt.legend()
    plt.savefig(outdir + 'loss_epochs.pdf'); plt.close()

    print("\033[92mLoss vs epochs plot saved to {}.\033[0m".format(outdir))

if __name__ == '__main__':
    main()
