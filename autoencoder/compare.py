import numpy as np
import matplotlib.pyplot as plt

# Takes as input two NP 2-dimensional arrays of the same size.

def compare(original, final):

	# MSE Computation.

	errors = [];

	rows = original.shape[0];
	cols = original.shape[1];

	for i in range(rows):
		err = np.mean((original[i] - final[i])**2);
		errors.append(err);

	print("MSE =", np.mean(errors))

	## PARAMETERS ##
	cool_variable = 1;
	divisions = 10;

	# Histogram.

	arr_original = original[:, cool_variable];
	arr_final = final[:, cool_variable];

	min_datum = min(np.min(original), np.min(final));
	max_datum = max(np.max(original), np.max(final));
	limits = np.arange(min_datum, max_datum, 1/divisions);

	plt.subplot(2,1,1)
	plt.hist(arr_original, bins = limits)
	plt.hist(arr_final, bins = limits)

	# Ratios.

	count_original = [];
	count_final = [];

	for i in range(len(limits) - 1):
		count_original.append(np.sum(
			(arr_original > limits[i]) & (arr_original < limits[i + 1])
		));
		count_final.append(np.sum(
			(arr_final > limits[i]) & (arr_final < limits[i + 1])
		));

	ratios = np.array(count_final) / np.array(count_original);
	limit_tags = limits[0:-1]

	plt.subplot(2,1,2)
	plt.scatter(limit_tags, ratios);
	plt.show();



