#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --mem=3000M
#SBATCH -o ./autoencoder_pytorch/logs/ae_train_%j.log

# Arguments:
# $1 : The training .npy file.
# $2 : The validation .npy file.
# $3 : The learning rate.
# $4 : The training batches.
# $5 : The number of epochs to train.
# $6 : A str flag to flag the file.

source /work/deodagiu/miniconda3/bin/activate qml_project
python3 main.py --training_file /work/deodagiu/qml_data/$1 --validation_file /work/deodagiu/qml_data/$2 --lr $3 --batch $4 --epochs $5 --maxdata_train $6 --file_flag $7
