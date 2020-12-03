import numpy as np
import matplotlib.pyplot as plt
import os
from varname import varname

# Takes as input two NP 2-dimensional arrays of the same size.

# If name or write are provided, assumes the existence of the out/ directory.
def compare(original1, final1, encoded1, original2, final2, encoded2, name = None, writeMSE = False):
	
	original1 = np.array(original1);
	final1 = np.array(final1);
	encoded1 = np.array(encoded1);
	original2 = np.array(original2);
	final2 = np.array(final2);
	encoded2 = np.array(encoded2);
	
	original = np.vstack((original1, original2))
	final = np.vstack((final1, final2))

	# MSE Computation.
	
	errors = [];
	
	rows = original.shape[0];
	cols = original.shape[1];

	enc_cols = min(encoded1.shape[1], encoded2.shape[1]);

	for i in range(rows):
		err = np.mean((original[i] - final[i])**2);
		errors.append(err);
	
	mse = np.mean(errors);

	if writeMSE:
		logf = open("out/MSELOG", "a")
		logf.write(str(mse) + " " + name + "\n") 
		logf.close()
	else:
		print("MSE =", mse)


	## PARAMETERS ##
	alph = 0.8;
	
	# Histogram.

	if (name != None):
		dirname = "out/" + name;
	
		for cool_variable in range(cols):
			divisions = 100;
			arr_original1 = original1[:, cool_variable];
			arr_final1 = final1[:, cool_variable];
			arr_original2 = original2[:,cool_variable];
			arr_final2 = final2[:,cool_variable];
	
			min_datum = min(np.min(original), np.min(final));
			max_datum = max(np.max(original), np.max(final));
	
			limits = np.arange(min_datum, max_datum, 1/divisions);
			limits = np.append(limits,max_datum+1/(10 * divisions))
	
			plt.figure();
			plt.hist(arr_original1, bins = limits,alpha = alph,histtype='step',linewidth=2.5,label="Background")	
			plt.hist(arr_original2, bins = limits,alpha = alph,histtype='step',linewidth=2.5,label="Signal")
			plt.hist(arr_final1, bins = limits, alpha = alph,histtype='step',linewidth=2.5, label = "AFTER Background")
			plt.hist(arr_final2, bins = limits, alpha = alph,histtype='step',linewidth=2.5, label = "AFTER Signal")
	
			plt.xlim(min_datum, max_datum)
			plt.legend();
			plt.title(varname(cool_variable) + " (" + name + ")")
			plt.savefig(dirname + '/hist' + str(cool_variable) + '.png');	
		

		for cool_variable in range(enc_cols):
			divisions = 50;
			arr_enc1 = encoded1[:, cool_variable];
			arr_enc2 = encoded2[:, cool_variable];
			min_datum = min(np.min(arr_enc1), np.min(arr_enc2));
			max_datum = max(np.max(arr_enc1), np.max(arr_enc2));

			limits = np.arange(min_datum, max_datum, (max_datum - min_datum)/divisions);
			limits = np.append(limits,max_datum+1/(10 * divisions))

			plt.figure();
			plt.hist(arr_enc1, bins = limits,alpha = alph,histtype='step',linewidth=2.5,label="Background")	
			plt.hist(arr_enc2, bins = limits,alpha = alph,histtype='step',linewidth=2.5,label="Signal")
			plt.xlim(min_datum, max_datum)
			plt.legend();
			plt.title("Encoded " + str(cool_variable) + " (" + name + ")")
			plt.savefig(dirname + '/enc' + str(cool_variable) + '.png');	




		
