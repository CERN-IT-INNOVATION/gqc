import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

import model_vasilis
import model_vasilis_tanh
import util
from util import compute_model

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--valid_file", type=str,
    help="The path to the validation data.")
parser.add_argument("--test_file", type=str,
    help="The path to the test data.")
parser.add_argument("--test_target", type=str,
    help="The path to the test target numpy file.")
parser.add_argument('--model_path', type=str, required=True,
    help='The path to the saved model.')

def main():
    args = parser.parse_args()
    device = 'cpu'

    # Import the hyperparameters and the data.
    layers, batch, lr = import_hyperparams(args.model_path)
    valid_data   = np.load(args.valid_file)
    test_data    = np.load(args.test_file)
    valid_loader = util.to_pytorch_data(valid_data, device, None, False)
    test_loader  = util.to_pytorch_data(test_data, device, None, False)

    test_data   = np.load(args.test_file)
    test_target = np.load(args.test_target)
    data_sig, data_bkg = util.split_sig_bkg(test_data, test_target)

    test_loader_sig = util.to_pytorch_data(data_sig, device, None, False)
    test_loader_bkg = util.to_pytorch_data(data_bkg, device, None, False)

    # Import the model.
    (layers).insert(0, test_data.shape[1])
    model = util.load_model(model_vasilis_improved.AE, layers, lr,
        args.model_path, device)
    criterion = model.criterion()

    # Compute criterion scores for data.
    compute_save_scores(args.model_path, model, criterion, valid_loader,
        test_loader)

    # Load the signal and background latent spaces.
    output_sig, sig_latent, input_sig = compute_model(model, test_loader_sig)
    output_bkg, bkg_latent, input_bkg = compute_model(model, test_loader_bkg)
    sig_latent = sig_latent.numpy()
    bkg_latent = bkg_latent.numpy()

    # Do the plots.
    latent_space_plots(sig_latent, bkg_latent, args.model_path)
    sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg,args.model_path)

    # Load sig/bkg data, but this time with batch_size.
    loader_sig = util.to_pytorch_data(data_sig, device, batch, True)
    loader_bkg = util.to_pytorch_data(data_bkg, device, batch, True)

    feature_nb = test_data.shape[1]
    sig_batch_loss = util.mean_batch_loss(loader_sig, model, feature_nb,
        criterion, device)
    bkg_batch_loss = util.mean_batch_loss(loader_bkg, model, feature_nb,
        criterion, device)

    mean_losses = [sig_batch_loss, bkg_batch_loss]
    batch_loss_plots(mean_losses, len(test_data), args.model_path)

def import_hyperparams(model_path):
    """
    Imports the hyperparameters of a given model from the model path given by
    the user.

    @model_path :: Path to the trained model folder of the autoencoder that
        you are plotting for.

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
    print("Layers: ", layers)
    print("Batch: ", batch)
    print("Learning Rate: ", lr)

    return layers, batch, lr

def vasilis_reprod_data():
    # Reproduces the way Vasilis normalizes data such that it leads to
    # identical plots.

    data_sig = np.load('/work/deodagiu/qml_data/input_ae/x_data_sig.npy')
    data_bkg = np.load('/work/deodagiu/qml_data/input_ae/x_data_bkg.npy')

    ntot = 360000
    data_sig = data_sig[:360000, :]
    data_bkg = data_bkg[:360000, :]
    all_data = np.vstack((data_sig, data_bkg))
    norm_data = MinMaxScaler().fit_transform(all_data)
    data_sig = norm_data[:360000, :]
    data_bkg = norm_data[360000:, :]

    ntrain, nvalid, ntest = int(0.8*ntot), int(0.1*ntot), int(0.1*ntot)
    data_sig = data_sig[ntrain+nvalid:ntot, :]
    data_bkg = data_bkg[ntrain+nvalid:ntot, :]

    return data_sig, data_bkg

def compute_score(model, loader, criterion, prepend_str, score_file):
    """
    Compute the score for a certain model and data.

    @model       :: The pytorch model object.
    @loader      :: The pytorch data loader object for the data.
    @criterion   :: The criterion object for the loss function.
    @prepend_str :: Descriptive string to prepend to output.

    @returns :: Writes to screen and to file the descriptive string and the
        computed loss function score.
    """
    output, latent, inp = compute_model(model, loader)
    print(prepend_str, criterion(output,inp).item())
    score_file.write(prepend_str + "{}\n".format(criterion(output,inp).item()))

def compute_save_scores(save_path, model, criterion, valid_data, test_data):
    # Computes the scores of the model on the test and train datasets and
    # saves the results to a file.
    print('\n----------------------------------')
    score_file = open(save_path + "score_file.txt", "w")
    compute_score(model, test_data, criterion, "TEST MSE:", score_file)
    compute_score(model, valid_data, criterion, "VALID MSE:", score_file)
    score_file.close()
    print('----------------------------------\n')

def ratio_plotter(input_data, output_data, ifeature, color, class_label=''):
    # Plots the overlaid input and output data to see how good the
    # reconstruction really is.
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
    # Makes the plots of the latent space data for each of the latent space
    # features, which should be 8 for the vasilis_model, for example.
    storage_folder_path = model_path + 'latent_plots/'
    if not os.path.exists(storage_folder_path):
        os.makedirs(storage_folder_path)
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

    print("\033[92mLatent plots were saved to {}.\033[0m"
        .format(storage_folder_path))

def sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg, model_path):
    # Plot the background and the signal distributions, overlaid.
    storage_folder_path = model_path + 'ratio_plots/'
    if not os.path.exists(storage_folder_path):
        os.makedirs(storage_folder_path)
    for idx in range(input_sig.numpy().shape[1]):
        plt.figure(figsize=(12,10))

        ratio_plotter(input_bkg.numpy()[:,idx], output_bkg.numpy()[:,idx], idx,
            'gray', class_label='Background')
        ratio_plotter(input_sig.numpy()[:,idx], output_sig.numpy()[:,idx], idx,
            'navy', class_label='Signal')

        plt.savefig(storage_folder_path + 'Ratio Plot ' +
            util.varname(idx) + '.pdf')
        plt.close()

    print("\033[92mRatio plots were saved to {}.\033[0m"
        .format(storage_folder_path))

def diagnosis_plots(loss_train, loss_valid, min_valid, nodes, batch_size, lr,
    epochs, outdir):
    # Quick plot to see if what we trained is of any good.
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

def batch_loss_plots(losses_sig_bkg, n_test, model_path):
    """
    Makes plot of the mean batch loss for the signal and background samples,
    overlays them and saves the final result.

    @losses_sig_bkg :: Array with two elements: mean batch losses array for
        the signal sample and the same for the bkg sample.
    @n_test         :: Number of test data events.
    @model_path     :: The path to the saved model.
    """
    colors = ['b', 'r']
    label  = ["Test on Sig.", "Test on Bkg."]
    for idx in range(len(losses_sig_bkg)):
        plt.hist(losses_sig_bkg[idx], bins=20, density=0, color=colors[idx],
            alpha=0.5, ec='black', label=label[idx])
        plt.ylabel('Entries/Bin')
        plt.xlabel('MSE per Batch')
        plt.title('MSE per batch, Ntest={}.'.format(n_test))

    plt.legend()
    plt.savefig(model_path + 'test_loss_hist.pdf', dpi=300)

if __name__ == '__main__':
    main()
