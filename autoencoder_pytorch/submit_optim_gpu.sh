#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/ae_optim_gpu_%j.log

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Input training file: $1"
echo "Input validation file: $2"
echo "Learning rate min: $3"
echo "Learning rate max: $4"
echo "Batch size min: $5"
echo "Batch size max: $6"
echo "Number of Epochs min: $7"
echo "Number of Epochs max: $8"
echo "Training data set size, i.e., events #: $9"
echo " "
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
python3 hyperparam_optimizer.py --training_file $1 --validation_file $2 --lr $3 $4 --batch $5 $6 --epochs $7 $8 --maxdata_train $9
