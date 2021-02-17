from sklearn.metrics import roc_curve
from qdata import qdata
import aeTF.encode as aetf
import matplotlib.pyplot as plt
import os

def get_roc(framework):
	folder = ""
	if framework == "tf":
		encode = aetf.encode
		folder = "aeTF/roc"
	elif framework == "pl":
		print("TODO") # TODO
	else:
		raise Exception("undefined framework")

	if not (os.path.isdir(folder)):
		os.mkdir(folder)

	qd = qdata(framework, 1, 1, 1)	

	features = qd.train.shape[1]

	for i in range(features):
		fpr, tpr, thres = roc_curve(qd.validation_nlabels, qd.validation[:,i])
		plt.figure();
		plt.plot(fpr, tpr)
		plt.title("ROC for " + str(i) + " " + framework)
		plt.savefig(folder + "/" + str(i) +  ".png");



