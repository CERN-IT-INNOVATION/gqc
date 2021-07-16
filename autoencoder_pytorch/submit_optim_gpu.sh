#!/bin/sh
#SBATCH --job-name=ae_optuna
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH -o ./logs/ae_optim_gpu_%j.log

usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>]" 1>&2; exit 1; }

while getopts ":n:s:l:b:e:" o; do
    case "${o}" in
    n)
        n=${OPTARG}
        ;;
    s)
        s=${OPTARG}
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

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Input training file: ${n}"
echo "Input validation file: ${s}"
echo "Learning rate min and max: ${l[@]}"
echo "Batch sizes: ${b[@]}"
echo "Number of Epochs: ${e}"
echo " "
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
export PYTHONUNBUFFERED=TRUE
python3 hyperparam_optimizer.py --data_folder /work/deodagiu/qml_data/input_ae/ --norm ${n} --nevents ${s} --lr ${l[@]}  --batch ${b[@]} --epochs ${e}
export PYTHONUNBUFFERED=FALSE
