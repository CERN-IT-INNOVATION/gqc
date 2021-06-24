# Autoencoder
The autoencoder framework used for feature reduction. At the moment of writing, there exists only one autoencoder architecture in model_vasilis.py. This autoencoder is a very basic one that reduces the number of features from, currently, 67 to 16. The resulting latent space features are passed on for training of the quantum classifiers (VQC and SVM). 

The workflow follows the diagram:
![preprocessing](autoencoder_workflow.png)

The normalised training data (with an adjustable total number of events that one can set) and the validation data are passed to main.py, which sets the hyper-parameters of the training and passes them further to the model itself. All the methods dealing with the training and evaluation of a model are in its python file. At the end of the main.py file execution, the output consists of two things: the best trained model in a pytorch file and a loss vs epochs figure, stored in the folder trained_models/\[model\_name\]. This best model is then imported in plotting.py, where the data is reconstructed from the latent space features and comparison plots are made. An abundance of plots are saved to the model folder in trained\_models as before.

TODO: Maybe softmax as output activation? To automatize the sum to 1 of amplitudes of the Qcircuit
