# Main script of the vqc.
# Imports the data for training. Imports the data for validation and testing
# and kfolds it into k=5.
# Computes the ROC curve of the qsvm and the AUC, saves the ROC plot.
from time import perf_counter
import numpy as np

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

from .vqc import VQC
from .terminal_colors import tcols
from . import qdata as qd
from . import util


seed = 12345
# Ensure same global behaviour.
algorithm_globals.random_seed = seed


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["model_path"],
        train_events=10000,
        valid_events=4000,
        test_events=6000,
        kfolds=0,
    )

    train_features = \
        qdata.batchify(qdata.get_latent_space("train"), args["batch_size"])
    train_labels = qdata.batchify(qdata.ae_data.trtarget, args["batch_size"])
    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget

    backend = Aer.get_backend("aer_simulator_statevector")
    qinst = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    vqc = VQC(args["nqubits"], args["feature_map"], args["ansatz"],
              train_features.shape[2], warm_start=False,
              quantum_instance=qinst, loss=args["loss"],
              optimizer=args["optimizer"])
    exit(1)

    print(tcols.OKCYAN + "Training the VQC..." + tcols.ENDC)
    util.print_model_info(args["model_path"], qdata, vqc)

    train_time_init = perf_counter()
    train(vqc, train_features, train_labels, args["epochs"])
    train_time_fina = perf_counter()
    print(f"Training completed in: {train_time_fina-train_time_init:.2e} s")

    train_acc = vqc.score(train_features, train_labels)
    valid_acc = vqc.score(valid_features, valid_labels)
    util.print_accuracies(valid_acc, train_acc)

    util.create_output_folder("trained_vqcs/" + args["output_folder"])
    util.save_vqc(vqc, "trained_vqcs/" + args["output_folder"] + "/model")


def train(vqc, train_features, train_labels, epochs):
    """
    Training the vqc.
    @vqc            :: The vqc qiskit object to be trained.
    @train_features :: Numpy array with training data divided into batches.
    @train_labels   :: Numpy array with training target div into batches.
    @epochs         :: Int of the number of epochs this should be trained.
    """
    for epoch in range(epochs):
        for data_batch, target_batch in zip(train_features, train_labels):
            vqc.fit(data_batch, target_batch)


def compute_model_scores(model, data_folds, output_folder) -> np.ndarray:
    """
    Computing the model scores on all the test data folds to construct
    performance metrics of the model, e.g., ROC curve and AUC.

    @model         :: The vqc model to compute the score for.
    @data_folds    :: Numpy array of kfolded data.
    @output_folder :: The folder where the results are saved.

    returns :: Array of the qsvm scores obtained.
    """
    scores_time_init = perf_counter()
    model_scores = np.array(
        [model.predict(fold) for fold in data_folds]
    )
    scores_time_fina = perf_counter()
    print(f"Completed in: {scores_time_fina-scores_time_init:2.2e} s")

    path = "trained_vqcs/" + output_folder + "/y_score_list.npy"

    print("Saving model scores array in: " + path)
    np.save(path, model_scores)

    return model_scores
