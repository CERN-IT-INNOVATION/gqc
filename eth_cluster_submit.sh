#!/bin/sh
# Arguments:
# $1 : The training .npy file.
# $2 : The validation .npy file.
# $3 : The learning rate.
# $4 : The training batches.
# $5 : The number of epochs to train.
# $6 : A str flag to flag the file.

#SBATCH --job-name=train_autoencoder                   
#SBATCH --account=gpu_gres        # to access gpu resources
#SBATCH --partition=gpu
#SBATCH --nodes=1                 # request to run job on single node                                       
#SBATCH --ntasks=5                # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1              # request 1 GPU's on machine
#SBATCH -o train_autoencoder.log
source /work/deodagiu/miniconda3/bin/activate qml_project
python3 autoencoder_pytorch/main.py --training_file /work/deodagiu/qml_data/$1 --validation_file /work/deodagiu/qml_data/$2 --lr $3 --batch $4 --epochs $5 --file_flag $6  
