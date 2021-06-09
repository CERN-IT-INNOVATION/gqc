# Uses the optuna library to optimize the hyperparameters for a given NN.
import argparse, os, time

import optuna
from optuna.trial import TrialState
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

import model_vasilis as basic_nn
import util
import plotting

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_file", type=str,
    help="The path to the training data.")
parser.add_argument("--valid_file", type=str,
    help="The path to the validation data.")
parser.add_argument('--lr', type=float, nargs=2,
    help='The learning rate range [min max].')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, nargs="+",
    help='The batch options, e.g., [128 256 512].')
parser.add_argument('--epochs', type=int,
    help='The number of training epochs.')
parser.add_argument('--maxdata_train', type=int, default=-1,
    help='The maximum number of training samples to use.')

def optuna_objective(trial):
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do.
    """
    args   = parser.parse_args()
    device = util.define_torch_device()

    # Define optuna parameters.
    lr     = trial.suggest_loguniform('lr', *args.lr)
    batch  = trial.suggest_categorical('batch', args.batch)

    # Load the data.
    train_loader, valid_loader = \
    util.get_train_data(args.train_file, args.valid_file, batch, device)

    # Define model.
    (args.layers).insert(0, np.load(args.train_file).shape[1])
    model = basic_nn.AE(nodes=args.layers,lr=args.lr,device=device).to(device)

    # Train model.
    start_time = time.time()
    loss_train, loss_valid, min_valid = model_vasilis.train(train_loader,
        valid_loader, model, device, args.epochs, None)
    end_time = time.time()
    train_time = (end_time - start_time)/60

    print("Training time: {:.2e} mins.".format(train_time))

    return min_valid

if __name__ == '__main__':
    # Start the optuna study.
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')

    study.optimize(optuna_objective, n_trials=10)

    comp_trials= study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(comp_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("\n\nParams: ")
    for key, value in trial.params.items(): print("{}: {}".format(key, value))
