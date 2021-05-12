#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --mem=3000M
#SBATCH -o ./logs/ae_train_%j.log

# Arguments:
# 1 : The training .npy file.
# 2 : The validation .npy file.
# 3 : The learning rate.
# 4 : The training batches.
# 5 : The number of epochs to train.
# 6 : The number of events to train into
# 7 : A str flag to flag the file.

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Input training file: $1"
echo "Input validation file: $2"
echo "Learning rate: $3"
echo "Batch size: $4"
echo "Number of Epochs: $5"
echo "Training data set size, i.e., events #: $6"
echo " "
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
python3 main.py --training_file /work/deodagiu/qml_data/$1 --validation_file /work/deodagiu/qml_data/$2 --lr $3 --batch $4 --epochs $5 --maxdata_train $6 --file_flag $7
