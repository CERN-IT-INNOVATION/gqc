import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from qdata import qdata


def get_info(name):
	qd = qdata()

	truelabels = qd.validation_nlabels

	data = "vqctf/out/" + name + ".npy"

	predictions = np.load(data)

	if len(predictions.shape) == 1:
		predictions = np.hsplit(predictions, 5)
	
	aucs = [];

	all_scores = []
	all_labels = []


	for i in range(len(predictions)):
		all_scores = np.concatenate((all_scores, predictions[i]))
		all_labels = np.concatenate((all_labels, truelabels))
		sample = predictions[i]
		results = []
		for j in range(len(sample)):
			if (sample[j] < 0.5):
				results.append(0)
			else:
				results.append(1)
		results = np.array(results)
		print(sum(results == truelabels) / len(results))

		aucs.append(roc_auc_score(truelabels, predictions[i]))

	print(aucs)
	print("AUC: " + str(np.mean(aucs)) + "+-" + str(np.std(aucs)))

	X = roc_curve(all_labels, all_scores)

	return [X[0], X[1], np.mean(aucs), np.std(aucs)]


def get_plot(model_dictionary,ntrain,nvalid=720,nfolds=5):
	
	qd = qdata()

	# Style config
	f1 = plt.figure(1,figsize=(10,10))
	plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
	plt.rc('axes', titlesize=20)     # fontsize of the axes title
	plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
	plt.rc('legend', fontsize=20)    # legend fontsize
	plt.rc('figure', titlesize=20)  # fontsize of the figure title


	for model_name in model_dictionary:
		X = model_dictionary[model_name]
		plt.plot(X[0],X[1],label = model_name+fr': AUC = {X[2]:.4f} $\pm$ {X[3]:.4f}')

	plt.title(r'$N^{train}$'+f'={ntrain},'+r' $N^{test}$'+f'={nvalid} ($x {nfolds}$)', loc='left')
	plt.xlabel('Background Efficiency (FPR)')
	plt.ylabel('Signal Efficiency (TPR)')

	x = np.linspace(0,1,num=50)#draw x=y line for random binary classifier ROC
	plt.plot(x,x,'--',color = 'k',label = 'Random Classifier')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.tight_layout()
	plt.legend()
	#f1.savefig('rocky.png')
	plt.show()



model_dictionary = {"VQC": get_info("AZ1-nenc-1"),
"RF": get_info("randomforest"),
"Log Reg": get_info("logreg"),
"AdaBoost": get_info("adaboost")}

#"MLP": get_info("mlperceptron"),


get_plot(model_dictionary,3000)
