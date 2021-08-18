from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector # Amplitude encoding.
#Custom feature maps:
from qsvm.feature_map_circuits import customFeatureMap,get_circuit14,u2Reuploading
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
import numpy as np
import time,argparse
import matplotlib.pyplot as plt
from sklearn import metrics

import qdata as qd


def main():
    #TODO: merge with test.py & have uncertainties on curves and AUC
    start_time = time.time()

    feature_map_dict = {'amp_enc_only':RawFeatureVector(2**4),
        'amp_enc_only6':RawFeatureVector(2**6),
        'u2Reuploading':u2Reuploading(nqubits = 8, nfeatures=16)}

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)#include defaults in -h
    parser.add_argument('--model',type = str,required = True,
                        help='Path to saved model excluding .npz')
    parser.add_argument('--feature_map',required = True, 
            help = f'Choose from available feature maps: {feature_map_dict.keys()}')
    parser.add_argument('--noEncoding', action = 'store_true', 
            help = 'If activated, the dataset will be used directly instead of latent space')
    parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
    parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
    parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
    parser.add_argument('--model_path', type=str, required=True,
    help="The path to the saved model.")
    args = parser.parse_args()

    qdata_loader = qd.qdata(args.data_folder, args.norm, args.nevents,
	 	args.model_path, train_events=576, valid_events=720, test_events=6, kfolds=5)

    backend = Aer.get_backend('statevector_simulator')
    qi = QuantumInstance(backend)
    feature_map = feature_map_dict[args.feature_map]
    qsvm = QSVM(feature_map,quantum_instance = qi)

    qsvm.load_model(args.model+'.npz')
    test_folds = qdata_loader.get_kfolded_data('test')

    y_scores = np.array([qsvm.instance.get_predicted_confidence(fold) for fold
                 in test_folds])
    np.save(args.model+'_yscoreList.npy',y_scores)

    end_time = time.time()

    models_names_dict = {'QSVM (4 qubits)': y_scores}
    get_plots(models_names_dict, 'giga_shark', qdata_loader)

    print(f'Total Runtime = {(end_time-start_time)/60:.4f} min.')

def get_plots(model_dictionary, filename, qdata_loader):
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
		auc = np.array([metrics.roc_auc_score(qdata_loader.ae_data.test_target, y_score) 
            for y_score in y_scores])
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
	f1.savefig(filename + ".pdf")
	plt.close()

if __name__ == '__main__':
    main()
