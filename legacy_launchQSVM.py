from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
from qsvm.feature_map_circuits import customFeatureMap,get_circuit14,u2Reuploading
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
import time, argparse
import qdata as qd
from datetime import datetime
import warnings
import os
import matplotlib.pyplot as plt
from sklearn import metrics
warnings.filterwarnings('ignore', category=DeprecationWarning)#for aqua


seed = 12345

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
parser.add_argument('--model_path', type=str, required=True,
    help="The path to the saved model.")
parser.add_argument('--output_folder', required = True,
    help='Model name to be saved.')

args = parser.parse_args()


def main():
    start_time  = time.time()
    feature_dim = 16
    qdata_loader = qd.qdata(args.data_folder, args.norm, args.nevents,
         args.model_path, train_events=10, valid_events=10, test_events=10,
         kfolds=5)

    if not os.path.exists('qsvm_output/' + args.output_folder):
    	os.makedirs('qsvm_output/' + args.output_folder)

    train_features = qdata_loader.get_latent_space('train')
    train_labels   = qdata_loader.ae_data.train_target
    valid_features = qdata_loader.get_latent_space('valid')
    valid_labels   = qdata_loader.ae_data.valid_target
    test_folds     = qdata_loader.get_kfolded_data('test')

    feature_map = RawFeatureVector(feature_dim)
    backend = QuantumInstance(Aer.get_backend('statevector_simulator'),
        seed_simulator=seed, seed_transpiler=seed)

    qsvm = QSVM(feature_map, quantum_instance=backend, lambda2=5)
    qsvm.train(train_features, train_labels)

    # Compute the accuracies for quick probe for overtraining
    valid_accuracy = qsvm.test(valid_features, valid_labels)
    train_accuracy = qsvm.test(train_features, train_labels)

    print(f'Test Accuracy = {valid_accuracy}')
    print(f'Training Accuracy = {train_accuracy}')

    end_time = time.time()
    runtime = end_time-start_time
    print(f'Total runtime: {runtime:.2f} sec.')

    save_model_log(qdata_loader, qsvm, runtime, train_accuracy, valid_accuracy)

    y_scores = np.array([qsvm.instance.get_predicted_confidence(fold) for fold
        in test_folds])
    np.save(args.output_folder + 'y_score_list.npy', y_scores)
    models_names_dict = {'QSVM (4 qubits)': y_scores}
    get_plots(models_names_dict, qdata_loader)


def save_model_log(qdata_loader, qsvm_model, runtime, train_accuracy,
    valid_accuracy):
    print(f'\n---------------------{datetime.now()}----------------------')
    print('Autoencoder model path:', args.model_path)
    print('Data path:', qdata_loader.ae_data.data_folder)
    print(f'ntrain = {len(qdata_loader.ae_data.train_target)}, '
            f'ntest = {len(qdata_loader.ae_data.valid_target)}, '
            f'lambda2 = {qsvm_model.lambda2}')
    print(f'Execution Time {runtime} s or {runtime/60} min.')
    print(f'Test Accuracy: {valid_accuracy}, '
          f'Training Accuracy: {train_accuracy}')
    print('-------------------------------------------\n')

    qsvm_model.save_model('qsvm_output/' + args.output_folder + "/train_output")

def get_plots(model_dictionary, qdata_loader):
    f1 = plt.figure(1,figsize=(10,10))

    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.rc('axes', titlesize=22)     # fontsize of the axes title
    plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
    plt.rc('legend', fontsize=22)    # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title

    for model_name in model_dictionary.keys():
        y_scores = model_dictionary[model_name]

        #computation of auc +/- 1sigma
        auc = np.array([metrics.roc_auc_score(qdata_loader.ae_data.test_target, y_score) for y_score in y_scores])
        auc_mean,auc_std = np.mean(auc), np.std(auc)
        print()
        print("\n"+model_name+" AUC's: \n", auc)
        print(f'AUC (mean) = {auc_mean} +/- {auc_std}')
        y_scores_flat = y_scores.flatten()

        fpr,tpr,_ = metrics.roc_curve(np.tile(qdata_loader.ae_data.test_target,qdata_loader.kfolds),
                y_scores_flat)
        plt.plot(fpr,tpr,label = model_name+fr': AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

    plt.title(r'$N^{train}$'+f'={qdata_loader.ntrain},'+
        r' $N^{test}$'+f'={qdata_loader.ntest} ($x 5$)', loc='left')
    plt.xlabel('Background Efficiency (FPR)')
    plt.ylabel('Signal Efficiency (TPR)')

    x = np.linspace(0,1,num=50)#draw x=y line for random binary classifier ROC
    plt.plot(x,x,'--',color = 'k',label = 'Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.legend()
    f1.savefig('qsvm_output/' + args.output_folder + "/roc_plot.pdf")
    plt.close()

if __name__ == '__main__':
    main()
