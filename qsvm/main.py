# Main script of the qsvm.
# Imports the data for training. Imports the data for validation and testing
# and kfolds it into k=5.
# Computes the ROC curve of the qsvm and the AUC, saves the ROC plot.
import warnings
from time import perf_counter
import numpy as np

from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
#from qiskit.circuit import ParameterVector

from sklearn.svm import SVC

from .terminal_colors import tcols
from . import qdata as qd
from . import util
from .feature_map_circuits import u2Reuploading
from . import plot

#TODO good way to import any backend required (mock, noise model, or real)
# without having if-statement imports.

# Warnings are suppressed since qiskit aqua obfuscates the output of this
# script otherwise (IBM's fault not ours.)
warnings.filterwarnings("ignore", category=DeprecationWarning)

seed = 12345
# Ensure same global behaviour.
algorithm_globals.random_seed = seed


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["model_path"],
        train_events=600,
        valid_events=720,
        test_events=720,
        kfolds=5,
    )

    train_features = qdata.get_latent_space("train")
    train_labels = qdata.ae_data.trtarget
    test_features = qdata.get_latent_space("test")
    test_labels = qdata.ae_data.tetarget
    test_folds = qdata.get_kfolded_data("test")

    feature_map = u2Reuploading(nqubits=8, nfeatures=args["feature_dim"])
    #Virtual to physical qubits, ordering is from 0->(n_qubits-1)
    initial_layout = [9,8,11,14,16,19,22,25]
    #TODO make the config adjustable from  argparse
    config = {'seed_transpiler':seed, 'seed_simulator':seed ,
              'optimization_level':3, 'initial_layout':initial_layout,
              'shots':5000}
    quantum_instance = util.configure_quantum_instance(
        ibmq_token=args["ibmq_token"],
        sim_type = args["sim_type"],
        backend_name= args["backend_name"],
        **config
    )
    from qiskit.visualization import plot_circuit_layout
    print(quantum_instance)
    print(quantum_instance.transpile(feature_map)[0].draw(
          output='text'))
    kernel = QuantumKernel(feature_map=feature_map, 
                           quantum_instance=quantum_instance)
 '''
    feature_map_params = ParameterVector(
        "par_x", 
        self._feature_map.num_parameters
        )
    qc_kernel = kernel.construct_circuit()
   #quantum_kernel_matrix = kernel.evaluate(x_vec = train_features)
    

    qsvm = SVC(kernel=kernel, C=args["c_param"])

    print(tcols.OKCYAN + "Training the QSVM..." + tcols.ENDC)
    util.print_model_info(args["model_path"], qdata, qsvm)

    train_time_init = perf_counter()
    qsvm.fit(train_features, train_labels)
    train_time_fina = perf_counter()
    print(f"Training completed in: {train_time_fina-train_time_init:.2e} s")

    train_acc = qsvm.score(train_features, train_labels)
    test_acc = qsvm.score(test_features, test_labels)
    util.print_accuracies(test_acc, train_acc)

    args["output_folder"] = args["output_folder"] + f"_c={qsvm.C}"
    util.create_output_folder(args["output_folder"])
    util.save_qsvm(qsvm, "qsvm_models/" + args["output_folder"] + "/model")

    print(tcols.OKCYAN +
          "\n\nComputing accuracies on kfolded test data..." +
          tcols.ENDC)
    scores = compute_model_scores(qsvm, test_folds, args["output_folder"])

    print(tcols.OKCYAN + "\n\nPlotting and saving ROC figure..." + tcols.ENDC)
    plot.roc_plot(scores, qdata, args["output_folder"], args["display_name"])
'''

def compute_model_scores(model, data_folds, output_folder) -> np.ndarray:
    """
    Computing the model scores on all the test data folds to construct
    performance metrics of the model, e.g., ROC curve and AUC.

    @model         :: The qsvm model to compute the score for.
    @data_folds    :: Numpy array of kfolded data.
    @output_folder :: The folder where the results are saved.

    returns :: Array of the qsvm scores obtained.
    """
    scores_time_init = perf_counter()
    model_scores = np.array(
        [model.decision_function(fold) for fold in data_folds]
    )
    scores_time_fina = perf_counter()
    print(f"Completed in: {scores_time_fina-scores_time_init:2.2e} s")

    path = "qsvm_models/" + output_folder + "/y_score_list.npy"

    print("Saving model scores array in: " + path)
    np.save(path, model_scores)

    return model_scores