# Uses the optuna library to optimize the hyperparameters for a given AE.
import argparse, os, time

import optuna
from optuna.trial import TrialState
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

import util

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
parser.add_argument("--aetype", type=str,
    help="The type of autoencoder that you will use, i.e., vanilla etc..")
parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
parser.add_argument('--lr', type=float, nargs=2,
    help='The learning rate range [min max].')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, nargs="+",
    help='The batch options, e.g., [128 256 512].')
parser.add_argument('--epochs', type=int,
    help='The number of training epochs.')

def optuna_train(train_loader, valid_loader, model, epochs, trial):
    # Training method for the autoencoder that was defined above.
    print('\033[96mTraining the AE model...\033[0m')

    for epoch in range(epochs):
        model.train()

        train_loss = model.train_all_batches(train_loader)
        valid_loss = model.valid(valid_loader, None)

        model.print_losses(epoch, epochs, train_loss.item(), valid_loss.item())

        trial.report(train_loss.item(), epoch)
        if trial.should_prune():         raise optuna.TrialPruned()
        if model.best_valid_loss > 0.09: raise optuna.TrialPruned()

    return model.best_valid_loss

def optuna_objective(trial):
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do.
    """
    args               = parser.parse_args()
    device             = util.define_torch_device()
    encoder_activation = nn.Tanh()
    decoder_activation = nn.Tanh()

    # Define parameters to be optimized by optuna.
    lr    = trial.suggest_loguniform('lr', *args.lr)
    batch = trial.suggest_categorical(' batch', args.batch)

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
        util.to_pytorch_data(train_data, train_target, device, batch, True)
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

    model = util.choose_ae_model(args.aetype, device, args.layers, lr,
        encoder_activation, decoder_activation)

    min_valid = \
        optuna_train(train_loader, valid_loader, model, args.epochs, trial)

    return min_valid

if __name__ == '__main__':
    # Start the optuna study.
    print('\n')
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="batch-lr-optimizer",
        sampler=sampler, direction='minimize',
        pruner=optuna.pruners.HyperbandPruner())

    study.optimize(optuna_objective, n_trials=2)

    comp_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(comp_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("\n\nParams: ")
    for key, value in trial.params.items(): print("{}: {}".format(key, value))
