# Uses the optuna library to optimize the hyperparameters for a given NN.
import argparse, os, time

import optuna
from optuna.trial import TrialState
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

import model_vasilis
import util
import plotting

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--training_file", type=str,
    help="The path to the training data.")
parser.add_argument("--validation_file", type=str,
    help="The path to the validation data.")
parser.add_argument('--lr', type=float, nargs=2,
    help='The learning rate range [min max].')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, nargs=2,
    help='The batch size range [min max].')
parser.add_argument('--epochs', type=int, nargs=2,
    help='The number of training epochs[min max].')
parser.add_argument('--maxdata_train', type=int, default=-1,
    help='The maximum number of training samples to use.')
parser.add_argument('--file_flag', type=str, default='',
    help='Flag the file in a certain way for easier labeling.')

def optuna_objective(trial):
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do.
    """
    args = parser.parse_args()
    device = util.define_torch_device()

    # Define optuna parameters.
    lr     = trial.suggest_loguniform('lr', args.lr[0], args.lr[1])
    batch  = trial.suggest_int('batch', args.batch[0], args.batch[1])
    epochs = trial.suggest_int('epochs', args.epochs[0], args.epochs[1])

    # Load the data.
    train_loader, valid_loader = util.get_train_data(args.training_file,
        args.validation_file, args.maxdata_train, batch, device)

    # Define model.
    (args.layers).insert(0, np.load(args.training_file).shape[1])
    model = model_vasilis.AE(node_number=args.layers, lr=lr).to(device)

    # Train model.
    start_time = time.time()
    loss_train, loss_valid, min_valid = model_vasilis.train(train_loader,
        valid_loader, model, device, epochs, None)
    end_time = time.time()

    train_time = (end_time - start_time)/60
    print("Training time: {:.2e} mins.".format(train_time), flush=True)

    plotting.diagnosis_plots(loss_train, loss_valid, min_valid,
        model.node_number, args.batch, args.lr, args.epochs, outdir)
    return min_valid

if __name__ == '__main__':
    # Start the optuna study.
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')

    study.optimize(optuna_objective, n_trials=20)

    comp_trials= study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(comp_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("\n\nParams: ")
    for key, value in trial.params.items(): print("{}: {}".format(key, value))
