#!/bin/sh
#SBATCH --job-name=ae_optuna
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH -o ./logs/ae_optim_gpu_%j.log

usage() { echo "Usage: $0 [-t <training_file_path>] [-v <validation_file_path>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>]" 1>&2; exit 1; }

while getopts ":t:v:l:b:e:s:" o; do
    case "${o}" in
    t)
        t=${OPTARG}
        ;;
    v)
        v=${OPTARG}
        ;;
    l)
        set -f
        IFS=' '
        l=(${OPTARG})
        ;;
    b)
        set -f
        IFS=' '
        b=(${OPTARG})
        ;;
    e)
        e=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${t}" ] || [ -z "${v}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Input training file: ${t}"
echo "Input validation file: ${v}"
echo "Learning rate min and max: ${l[@]}"
echo "Batch sizes: ${b[@]}"
echo "Number of Epochs: ${e}"
echo " "
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
export PYTHONUNBUFFERED=TRUE
python3 hyperparam_optimizer.py --train_file /work/deodagiu/qml_data/input_ae/${t} --valid_file /work/deodagiu/qml_data/input_ae/${v} --lr ${l[@]}  --batch ${b[@]} --epochs ${e}
export PYTHONUNBUFFERED=FALSE
