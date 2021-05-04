# Autoencoder
The autoencoder framework used for feature reduction. At the moment of writing, there exists only one autoencoder architecture, stored in model_vasilis.py. This autoencoder is a very basic one that reduces the number of features from, currently, 67 to 8. The resulting feature arrays are passed on for training of the quantum classifiers (VQC and SVM).








TODO: Maybe softmax as output activation? To automatize the sum to 1 of amplitudes of the Qcircuit
