import time,  argparse, sys, os, warnings, joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel

from sklearn.svm import SVC
from sklearn import metrics


import qdata as qd
from qsvm.feature_map_circuits import u2Reuploading
warnings.filterwarnings('ignore', category=DeprecationWarning)#for aqua

seed = 12345
algorithm_globals.random_seed = seed # ensure same global behaviour

#TODO: Include hyperparameter optimisation for C (maybe splitting of 
# is needed)

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
	default = '../qml_data/input_ae/', 
	help = "The folder where the data is stored on the system.")
parser.add_argument("--norm", type=str, default='minmax',
    help = "The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str, default='7.20e+05',
    help = "The number of events of the norm file.")
parser.add_argument('--model_path', type=str, required = True, 
	help = "The path to the Auto-Encoder model.")
parser.add_argument('--output_folder', required = True,
	help = 'The name of the model to be saved.')
	#output_folder to follow some expressive convention including model info
	#like the AE implementation.
parser.add_argument('--display_name', type = str, default = 'QSVM (8 qubits)', 
	help = 'QSVM display name on the ROC plot.')

args = parser.parse_args()


def main():
	start_time = time.time()
	feature_dim=16
	qdata_loader = qd.qdata(args.data_folder, args.norm, args.nevents,
	args.model_path, train_events=6, valid_events=6, test_events=6, kfolds = 5)

	if not os.path.exists('qsvm/trained_models/' + args.output_folder):
		os.makedirs('qsvm/trained_models/' + args.output_folder)

	train_features = qdata_loader.get_latent_space('train')
	train_labels   = qdata_loader.ae_data.train_target
	test_features  = qdata_loader.get_latent_space('test')
	test_labels    = qdata_loader.ae_data.test_target
	test_folds     = qdata_loader.get_kfolded_data('test')

	feature_map = u2Reuploading(nqubits=8, nfeatures=feature_dim)
	backend = Aer.get_backend('aer_simulator_statevector')
	quantum_instance = QuantumInstance(backend, seed_simulator=seed,
		 seed_transpiler = seed)
	quantum_kernel = QuantumKernel(feature_map = feature_map,
		quantum_instance = quantum_instance)
	
	qsvm = SVC(kernel = quantum_kernel.evaluate)
	qsvm.fit(train_features, train_labels)
	
	test_accuracy = qsvm.score(test_features, test_labels)
	train_accuracy = qsvm.score(train_features,train_labels)

	#Quick probe for overtraining
	print(f'Test Accuracy = {test_accuracy}')
	print(f'Training Accuracy = {train_accuracy}')

	end_time = time.time()
	runtime = end_time-start_time
	print(f'Total runtime: {runtime:.2f} sec.')
	
	save_model_log(qdata_loader, qsvm, runtime, train_accuracy, test_accuracy)
	
	y_scores = compute_model_scores(model = qsvm, data_folds = test_folds)
	models_names_dict = {args.display_name : y_scores}
	generate_plots(models_names_dict, qdata_loader)


def save_model_log(qdata_loader, qsvm_model, runtime, train_accuracy,
    test_accuracy):
	with open('qsvm/models_train.log', 'a+') as f:
		original_stdout = sys.stdout
		sys.stdout = f
		print(f'\n---------------------{datetime.now()}----------------------')
		print('QSVM model:', args.output_folder)
		print('Autoencoder model:', args.model_path)
		print('Data path:', qdata_loader.ae_data.data_folder)
		print(f'ntrain = {len(qdata_loader.ae_data.train_target)}, '
				f'ntest = {len(qdata_loader.ae_data.test_target)}, '
			 	f'C = {qsvm_model.C}')
		print(f'Execution Time {runtime:.2f} s or {(runtime/60):.2f} min.')
		print(f'Test Accuracy: {test_accuracy}, Training Accuracy: {train_accuracy}')
		#TODO implement feature_map printing and/or name printing
		print('-------------------------------------------\n')
		sys.stdout = original_stdout

	save_qsvm(model = qsvm_model, path = 'qsvm/trained_models/' + args.output_folder
				+ '/qsvm_model')


def save_qsvm(model, path):
	'''
	To save sklearn models joblib package is used. Serialization and 
	de-serialization of objects is python-version sensitive.

	Note on alternatives: As of Python 3.8 and numpy 1.16, pickle protocol 5 
	introduced in PEP 574 supports efficient serialization and de-serialization
	for large data buffers natively using the standard library:
		 pickle.dump(large_object, fileobj, protocol=5)
	'''
	joblib.dump(model, path)
	print('Trained model saved in: ' + path +'\n')

def load_qsvm(path):
	'''Load model from pickle file, i.e., deserialisation.'''
	return joblib.load(path)

def compute_model_scores(model, data_folds) -> np.ndarray:
	'''Computing the model scores on all the test data folds to construct
	performance metrics of the model, e.g., ROC curve and AUC.'''
	
	print('Computing model scores on the test data folds...')
	model_scores = np.array([model.decision_function(fold) for fold in data_folds])
	
	path = 'qsvm/trained_models/' + args.output_folder + '/y_score_list.npy'
	print('Saving model scores array in: ' + path)
	np.save(path, model_scores)
	
	return model_scores

def generate_plots(model_dictionary, qdata_loader):
	#TODO: Have the mean ROC (solid line) + the individual ROCs per fold (thiner line)
	# + 1 std band (transparent)
    f1 = plt.figure(1,figsize=(10,10))

    plt.rc('xtick', labelsize=20)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)   # fontsize of the tick labels
    plt.rc('axes', titlesize=22)    # fontsize of the axes title
    plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
    plt.rc('legend', fontsize=22)   # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title

    for model_name in model_dictionary.keys():
        y_scores = model_dictionary[model_name]
        #computation of auc +/- 1sigma
        auc = np.array([metrics.roc_auc_score(
				qdata_loader.ae_data.test_target, y_score) for y_score in y_scores])
        auc_mean, auc_std = np.mean(auc), np.std(auc)
        print("\n\n"+model_name+" AUC's: \n", auc)
        print(f'AUC (mean) = {auc_mean} +/- {auc_std}')
        y_scores_flat = y_scores.flatten()
        fpr,tpr,_ = metrics.roc_curve(np.tile(qdata_loader.ae_data.test_target,
					qdata_loader.kfolds), y_scores_flat)
        plt.plot(fpr,tpr,label = model_name+fr': AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

    plt.title(r'$N^{train}$'+f'={qdata_loader.ntrain},'+
        r' $N^{test}$'+f'={qdata_loader.ntest} ($x 5$)', loc='left')
    plt.xlabel('Background Efficiency (FPR)')
    plt.ylabel('Signal Efficiency (TPR)')

    x = np.linspace(0,1,num=50) #draw x=y line for random binary classifier ROC
    plt.plot(x,x,'--',color = 'k',label = 'Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.legend()
    f1.savefig('qsvm/trained_models/' + args.output_folder + "/roc_plot.pdf")
    plt.close()


if __name__ == '__main__':
	main()
