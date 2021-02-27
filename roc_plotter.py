import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from qdata import qdata as qd 

qdata = qd('pt')
models_names_dict_16 = {'QSVM (4 qubits)':'amp_enc_only_yscoreList.npy', 'SVM rbf':'svm_rbf_yscoreList.npy',
'QSVM (8 qubits)':'u2_reuploading_yscoreList.npy'}
#'SVM linear':'svm_linear_yscoreList.npy'<-lower auc

labels_flat = np.tile(qdata.validation_nlabels,5)#construct the labels for flattened y_score vector (5,720)->(3600,)
#ROC plots latent space:
f1 = plt.figure(1,figsize=(10,10))
plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
plt.rc('ytick', labelsize=17)    # fontsize of the tick labels
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title
for model_name in models_names_dict_16.keys():
    y_scores = np.load('qsvm/'+models_names_dict_16[model_name])
    #computation of auc +/- 1sigma
    auc = np.array([metrics.roc_auc_score(qdata.validation_nlabels,y_score) for y_score in y_scores])
    auc_mean,auc_std = np.mean(auc), np.std(auc)
    print()
    print("\n"+model_name+" AUC's: \n", auc)
    print(f'AUC (mean) = {auc_mean} +/- {auc_std}')
    y_scores_flat = y_scores.flatten()
    fpr,tpr,_ = metrics.roc_curve(labels_flat,y_scores_flat)
    plt.plot(fpr,tpr,label = model_name+fr': AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

plt.title(r'$N^{train}$'+f'={len(qdata.train)},'+r' $N^{test}$'+f'={len(qdata.validation)} ($x 5$)',
loc='left')
plt.xlabel('Background Efficiency (FPR)')
plt.ylabel('Signal Efficiency (TPR)')

x = np.linspace(0,1,num=50)#draw x=y line for random binary classifier ROC
plt.plot(x,x,'--',color = 'k',label = 'Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.legend()
f1.savefig('roc_ae_latent_space.png')

#Plots for input space models:
models_names_dict_input = { 'QSVM (6 qubits)':'amp_enc_noEnc64_yscoreList.npy',
'SVM linear':'svm_linear_yscoreListINPUT.npy'}
#'SVM INPUT rbf':'svm_rbf_yscoreListINPUT.npy'<- lower AUC.

f2 = plt.figure(2,figsize=(10,10))
for model_name in models_names_dict_input.keys():
    y_scores = np.load('qsvm/'+models_names_dict_input[model_name])
    #computation of auc +/- 1sigma
    auc = np.array([metrics.roc_auc_score(qdata.validation_nlabels,y_score) for y_score in y_scores])
    auc_mean,auc_std = np.mean(auc), np.std(auc)
    print()
    print("\n"+model_name+" AUC's: \n", auc)
    print(f'AUC (mean) = {auc_mean} +/- {auc_std}')
    y_scores_flat = y_scores.flatten()
    fpr,tpr,_ = metrics.roc_curve(labels_flat,y_scores_flat)
    plt.plot(fpr,tpr,label = model_name+fr': AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

plt.title(r'$N^{train}$'+f'={len(qdata.train)},'+r' $N^{test}$'+f'={len(qdata.validation)} ($x 5$)',
loc='left')
plt.xlabel('Background Efficiency (FPR)')
plt.ylabel('Signal Efficiency (TPR)')
x = np.linspace(0,1,num=50)#draw x=y line for random binary classifier ROC
plt.plot(x,x,'--',color = 'k',label = 'Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.legend()
f2.savefig('roc_input.png')


#Plots for AUC individual selection of 16 best features from the input
models_names_dict_16auc = {'QSVM (4 qubits)':'amp_enc_16feature_reduction_yscoreList.npy',
'SVM rbf':'svm_rbf_yscoreList16AUC.npy'}
#'SVM linear':'svm_linear_yscoreList16AUC.npy'<- lower auc

f3 = plt.figure(3,figsize=(10,10))
for model_name in models_names_dict_16auc.keys():
    y_scores = np.load('qsvm/'+models_names_dict_16auc[model_name])
    #computation of auc +/- 1sigma
    auc = np.array([metrics.roc_auc_score(qdata.validation_nlabels,y_score) for y_score in y_scores])
    auc_mean,auc_std = np.mean(auc), np.std(auc)
    print()
    print("\n"+model_name+" AUC's: \n", auc)
    print(f'AUC (mean) = {auc_mean} +/- {auc_std}')
    y_scores_flat = y_scores.flatten()
    fpr,tpr,_ = metrics.roc_curve(labels_flat,y_scores_flat)
    plt.plot(fpr,tpr,label = model_name+fr': AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

plt.title(r'$N^{train}$'+f'={len(qdata.train)},'+r' $N^{test}$'+f'={len(qdata.validation)} ($x 5$)',
loc='left')
plt.xlabel('Background Efficiency (FPR)')
plt.ylabel('Signal Efficiency (TPR)')

x = np.linspace(0,1,num=50)#draw x=y line for random binary classifier ROC
plt.plot(x,x,'--',color = 'k',label = 'Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.legend()
f3.savefig('roc_feature_selection16.png')

plt.show()
