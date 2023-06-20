# Grid search module for the hyperoptimisation of class NN model. Specifically,
# we hyperoptimise the learning rate and the batch size of the training.
# The training and testing of the networks occurs within this modules.

import sys

sys.path.append("..")
import argparse
from typing import List, Tuple

from vqc_pennylane.terminal_colors import tcols
import train
import test


def main():
    args_train, args_test, learning_rate_space, batch_size_space = get_arguments()
    original_outdir, best_perf = init_perf_eval(
        args_train, learning_rate_space, batch_size_space
    )
    final_best_perf = run_grid_search(
        args_train,
        args_test,
        learning_rate_space,
        batch_size_space,
        original_outdir,
        best_perf,
    )
    print_results(final_best_perf)


def print_results(final_best_perf: dict):
    """Prints the results of the grid search."""
    print("\n\n" + tcols.SPARKS + tcols.UNDERLINE, end="")
    print(" Best performing model " + tcols.ENDC + tcols.SPARKS)
    print(f"AUC: {final_best_perf['auc']:.3f} ± {final_best_perf['std']:.3f}")
    print(
        "Saved in: "
        + tcols.BOLD
        + f"{final_best_perf['model_folder']}/ "
        + tcols.ENDC
        + tcols.ROCKET
        + "\n"
    )


def run_grid_search(
    args_train: dict,
    args_test: dict,
    learning_rate_space: List[float],
    batch_size_space: List[int],
    original_outdir: str,
    best_perf: dict,
):
    """Executes the for loops for the grid search hyperparameter optimisation.

    Args:
        args_train: The arguments and hyperparamters dictionary for the train module.
        args_test: The arguments and hyperparamters dictionary for the test module.
        learning_rate_space: List of learning rate values that we are interested in
                             scanning via grid search.
        batch_size_space: List of batch size values that we are interested in
                             scanning via grid search.
        original_outdir: The initial folder name in which the trained model will be
                         saved. It is passed through argpars and will change during
                         training as defined in `update_hpars`.
        best_perf: Dictionary that stores the best AUC values along with their std
                   and folder in which the corresponding model is saved.

    Returns: The values that correspond to the best performing model in the best_perf
             dictionary format.
    """
    for batch_size in batch_size_space:
        for lr in learning_rate_space:
            print(
                tcols.BOLD + tcols.UNDERLINE + f"\nTraining and testing for "
                f"batch_size = {batch_size} & learning_rate = {lr}" + tcols.ENDC
            )
            update_hpars(args_train, args_test, batch_size, lr, original_outdir)
            train.main(args_train)
            auc = test.main(args_test)
            check_best_performance(auc[0], auc[1], best_perf, args_train["outdir"])
            write_log_file(original_outdir, auc, args_train["outdir"])
    return best_perf


def check_best_performance(auc: float, std: float, best_perf: dict, outdir: str):
    """Check best performing model according to its AUC value. Update the best
    performing model info in a dictionary that contains the AUC and its std along
    with the name of the folder that contains the best performing model.

    Args:
        auc: The AUC of the current model.
        std: The standard deviation of the model.
        best_perf: The dictionary that contains the previous best performing model
                   info in a form `{'auc': auc, 'std': std, 'outdir': outdir}.
    """
    if best_perf["auc"] < auc:
        print(tcols.OKBLUE + f"Found new best AUC: {auc:.3f} ± {std:.3f}" + tcols.ENDC)
        print(f"Current best model saved in {outdir}")
        best_perf.update({"auc": auc, "std": std, "model_folder": outdir})


def update_hpars(
    args_train: dict, args_test: dict, batch_size: int, lr: float, origin_outdir: str
) -> dict:
    """Update the hyperparameters of the NN classifier and prepare for the next
    train-test iteration.

    Args:
        args_train: Current arguments and hyperparameters dictionary
                    for the train module.
        args_test: Current arguments and hyperparameters dictionary
                    for the test module.
        batch_size: The batch size for the current grid search iteration.
        lr: The learning rate for the current grid search iteration.
        origin_outdir: Name of output folder of the NN training, needed to change
                       the output directory at every grid search iteration.

    Returns: The updated hyperparameter dictionary used by the train and test modules.
    """
    new_outdir = origin_outdir + f"_b{batch_size}_lr{lr}"
    new_hpars = {"lr": lr, "batch_size": batch_size, "outdir": new_outdir}
    args_train.update(new_hpars)
    args_test.update({"nn_model_path": "trained_nns/" + new_outdir + "/best_model.pt"})


def init_perf_eval(
    args_train: dict, learning_rate_space: List[float], batch_size_space: List[int]
) -> Tuple[str, dict]:
    """Declare and initialise the best performing model variables: AUC and std values
    and directory in which the best model is saved. These variables will be updated
    during the grid search to identify the best performing model.

    Args:
        args_train: The arguments and hyperparamters dictionary for the train module.
        learning_rate_space: List of learning rate values that we are interested in
                             scanning via grid search.
        batch_size_space: List of batch size values that we are interested in
                             scanning via grid search.

    Returns: The initial original_oudir and best_perf values.
    """
    original_outdir = args_train["outdir"]
    best_perf = {"auc": -999, "std": -999, "model_folder": None}
    print(
        tcols.BOLD + tcols.HEADER + "\nGrid search "
        f"for batch_size = {batch_size_space} and "
        f"learning_rate = {learning_rate_space}" + tcols.ENDC
    )
    write_log_file(original_outdir)
    return original_outdir, best_perf


def write_log_file(
    path: str, auc: Tuple[float, float] = None, model_outdir: str = None
):
    """Logs the AUC's, the corresponding std's, and the corresponding paths in which
    the models are saved. If a log file with the same name exists, it will be
    overwritten at the beggining of the grid search.

    Args:
        path: Name prefix of the directories in which the models trained and tested
              through grid search will be saved. E.g., giving `--outdir grid_search`
              with argparse will produce folders with the prefix `grid_search` of
              the form: `grid_search_b<batch_size>_lr<lr>`. The log file will be
              stored in `trained_nns/grid_search.log`.
        auc: The tuple that contains the AUC and its std for a given model.
        model_outdir: The name of the folder in which the given model is saved.

    """
    path = f"trained_nns/{path}.log"
    if auc is None and model_outdir is None:
        print(f"Creating logging file: {path}")
        with open(path, "w") as file:
            file.write("--- Initialising grid search ---\n\n")
    else:
        print("Appending results to log file.")
        with open(path, "a") as file:
            text = f"AUC: {auc[0]:.3f} ± {auc[1]:.3f} | {model_outdir}\n"
            file.write(text)


def get_arguments():
    """
    Parses the command line arguments for the train and test modules to be used in
    grid search.

    Returns: Two dictionaries for the train and test modules and two lists for the
             learning rate and batch size grid search.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="The folder where the data is stored on the system..",
    )
    parser.add_argument(
        "--norm", type=str, help="The name of the normalisation that you'll to use."
    )
    parser.add_argument(
        "--ae_model_path", type=str, help="The path to the Auto-Encoder model."
    )
    parser.add_argument(
        "--nevents", type=str, help="The number of signal events of the norm file."
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=-1,
        help="The exact number of training events used < nevents.",
    )
    parser.add_argument(
        "--nvalid",
        type=int,
        default=-1,
        help="The exact number of valid events used < nevents.",
    )
    parser.add_argument(
        "--ntest",
        type=int,
        default=-1,
        help="The exact number of testing events used < nevents.",
    )
    parser.add_argument("--lr", type=float, default=2e-03, help="The learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--epochs", type=int, default=85, help="The number of training epochs."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Flag the file in a certain way for easier labeling.",
    )
    parser.add_argument(
        "--kfolds", type=int, default=5, help="Number of folds for the test."
    )
    parser.add_argument(
        "--batch_size_space",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="The batch size space to be probed in the grid search.",
    )
    parser.add_argument(
        "--learning_rate_space",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.005, 0.01, 0, 1],
        help="The learning rate space to be probed in the grid search.",
    )
    args = parser.parse_args()

    seed = 12345
    args_train = {
        "data_folder": args.data_folder,
        "norm": args.norm,
        "nevents": args.nevents,
        "ae_model_path": args.ae_model_path,
        "ntrain": args.ntrain,
        "nvalid": args.nvalid,
        #"layers": [67, 64, 52, 44, 32, 24, 16,],
        #"layers": [60, 52, 44, 32, 24, 16,],
        "layers": [67, 16],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "out_activ": "nn.Sigmoid()",
        "adam_betas": (0.9, 0.999),
        "outdir": args.outdir,
        "seed": seed,
    }

    args_test = {
        "data_folder": args.data_folder,
        "norm": args.norm,
        "nevents": args.nevents,
        "ae_model_path": args.ae_model_path,
        "nn_model_path": "trained_nns/" + args_train["outdir"] + "/best_model.pt",
        "nvalid": args.nvalid,
        "ntest": args.ntest,
        "seed": seed,
        "kfolds": args.kfolds,
    }
    return args_train, args_test, args.learning_rate_space, args.batch_size_space


if __name__ == "__main__":
    main()
