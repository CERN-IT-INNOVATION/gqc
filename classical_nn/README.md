# Classical benchmark with a fully connected feed-forward neural network 
The modules in this folder define the neural network architecture, its training and testing procedures. The goal is to have a classical neural network with the same architecture as the encoder part of the quantum HybridVQC and HybridAE models and to test its classification performance on the same dataset.

## Usage
All the scripts are meant to be executed from within `classical_nn/`. The modules are isolated from the other packages.

For the hyperoptimisation of the network a grid search is used on the learning rate and the batch size, keeping the network architecture intact. The model that performs best in terms of AUROC (Area under the ROC curve) is the one used for the benchmark against the quantum hybrid models.

### Modules
- `neural_network.py`: Defines the fully-connected NN model class.
- `train.py`: Performs the training of the NN classifier. The saved model corresponds to the model that scored a minimum validation loss throughout the training. Can be individually ran: 
  ```
  python train.py --data_folder path/to/data/ --norm minmax --nevents 7.20e+05 --ae_model_path /path/to/saved_AE/best_model.pt --ntrain 12000 --nvalid 1500 --epochs 85 --outdir test_train --batch_size 1024 --lr 0.01
  ```
- `test.py`: Performs the evaluation of the model on a k-folded test dataset. That is, k test samples are prepared and on each of the folds the ROC curves along with the ROCAUC are computed. The mean and std of the ROC curves and the AUC are finally computed to investigate the performance of the model. To test a specific model:
  ```
  python test.py --data_folder path/to/data/ --norm minmax --nevents 7.20e+05 --ae_model_path /path/to/saved_AE/best_model.pt --nevents 7.20e+05 --nvalid 1500 --ntest 3600 --nn_model_path trained_nns/test_train/best_model.pt --kfolds 5
  ```
- `grid_search.py`: Performs a linear grid search for the hyperoptimisation of the model. The hyperparameter space that is scanned is simply that of learning rate and batch size. To execute the grid search you can for example run:
  ```
  python grid_search.py --data_folder path/to/data/ --norm minmax --nevents 7.20e+05 --ae_model_path /path/to/saved_AE/best_model.pt --ntrain 12000 --nvalid 1500 --epochs 85 --outdir test_grid_search --batch_size 1024 --lr 0.01 --ntest 3600 --batch_size_space 512 1024 --learning_rate_space 0.005 0.01
  ```
