import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from qdata import qdata as qd 

qdata = qd('pt')
labels_flat = np.tile(qdata.validation_nlabels,5)#construct the labels for flattened y_score vector (5,720)->(3600,)
#ROC plots latent space:

def get_plots(model_dictionary, filename):
	f1 = plt.figure(1,figsize=(10,10))

	plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
	plt.rc('axes', titlesize=22)     # fontsize of the axes title
	plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
	plt.rc('legend', fontsize=22)    # legend fontsize
	plt.rc('figure', titlesize=22)  # fontsize of the figure title

	for model_name in model_dictionary.keys():
		y_scores = np.load('qsvm/'+model_dictionary[model_name])
	 	#computation of auc +/- 1sigma
		auc = np.array([metrics.roc_auc_score(qdata.validation_nlabels,y_score) for y_score in y_scores])
		auc_mean,auc_std = np.mean(auc), np.std(auc)
		print()
		print("\n"+model_name+" AUC's: \n", auc)
		print(f'AUC (mean) = {auc_mean} +/- {auc_std}')
		y_scores_flat = y_scores.flatten()
		fpr,tpr,_ = metrics.roc_curve(labels_flat,y_scores_flat)
		plt.plot(fpr,tpr,label = model_name+fr': AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

	plt.title(r'$N^{train}$'+f'={len(qdata.train)},'+r' $N^{test}$'+f'={len(qdata.validation)} ($x 5$)', loc='left')
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


models_names_dict_16 = {'QSVM (4 qubits)':'amp_enc_only_yscoreList.npy', 'SVM rbf':'svm_rbf_yscoreList.npy',
'QSVM (8 qubits)':'u2_reuploading_yscoreList.npy'}
models_names_dict_input = { 'QSVM (6 qubits)':'amp_enc_noEnc64_yscoreList.npy',
'SVM linear':'svm_linear_yscoreListINPUT.npy'}
models_names_dict_16auc = {'QSVM (4 qubits)':'amp_enc_16feature_reduction_yscoreList.npy',
'SVM rbf':'svm_rbf_yscoreList16AUC.npy'}

get_plots(models_names_dict_16, "roc_ae_latent_space")
get_plots(models_names_dict_input, "roc_input")
get_plots(models_names_dict_16auc, "roc_feature_selection16")




