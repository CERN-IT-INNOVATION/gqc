# Classical fully connected feed-forward neural network benchmark
The modules in this folder define the neural network architecture and its training and testing procedures. The goal is to have a classical neural network with the same layers as the encoder part of the HybridVQC and to test its classification performance on the same dataset.

## Modules
- `train.py`: Performs the training of the NN classifier. The saved model corresponds to the model that scored a minimum validation loss throughout the training.
- `test.py`: Performs the evaluation of the model on a k-folded test dataset. That is, k test samples are prepared and on each of the folds the ROC curves along with the ROCAUC are computed. The mean and std of the ROC curves and the AUC are finally computed to investigate the performance of the model.
- `neural_network.py`: Defines the fully-connected NN model class.
- `grid_search.py`: Performs a linear grid search for the hyperoptimisation of the model. The hyperparameter space that is scanned is simply that of learning rate and batch size. The architecture of the classical NN model is kept the same in order to keep the same order of weights between the classical and quantum (hybrid) models. (_also the output activation is probed_?)