# Grid search module for the hyperoptimisation of class NN model. Specifically,
# we hyperoptimise the learning rate and the batch size of the training. 
# The training and testing of the networks occurs within this modules.

import sys
sys.path.append("..")
import argparse

from vqc_pennylane.terminal_colors import tcols
import train 
import test

def main():
    args_train, args_test, batch_size_space, learning_rate_space = get_arguments()

    print(tcols.HEADER + "Initialising grid search for {batch_size_space} and "
          f"{learning_rate_space}" + tcols.ENDC)
    for batch_size in batch_size_space:
        for lr in learning_rate_space:
            
            train.main(args_train)
            #args_test = get_test_args()
            #test.main(args_test)


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
    args = parser.parse_args()

    seed = 12345
    #batch_size_space = [128, 256, 512, 1024, 2048]
    #learning_rate_space = [0.0001, 0.001, 0.005, 0.01, 0,1]
    batch_size_space = [512, 1024]
    learning_rate_space = [0.005, 0.01]
    args_train = {
        "data_folder": args.data_folder,
        "norm": args.norm,
        "nevents": args.nevents,
        "ae_model_path": args.ae_model_path,
        "ntrain": args.ntrain,
        "nvalid": args.nvalid,
        "ae_layers": [67, 64, 52, 44, 32, 24, 16, 1],
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
    
    return args_train, args_test, learning_rate_space, batch_size_space

if __name__ == "__main__":

    main()