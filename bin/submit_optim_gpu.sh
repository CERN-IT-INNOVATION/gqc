#!/bin/sh
#SBATCH --job-name=ae_optuna
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --time=7-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1
#SBATCH --output=trained_models/logs/ae_optim_gpu_%j.log

usage() { echo "Usage: $0 [-n <normalization_name>] [-t <autencoder_type>] [-s <number_of_events>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>]" 1>&2; exit 1; }

while getopts ":n:t:s:l:b:e:" o; do
    case "${o}" in
    n)
        n=${OPTARG}
        ;;
    s)
        s=${OPTARG}
        ;;
    t)
	t=${OPTARG}
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

if [ -z "${n}" ] || [ -z "${t}" ] || [ -z "${s}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "The normalization of the data: ${n}"
echo "Type of autoencoder: ${t}"
echo "The number of events: ${s}"
echo "Learning rate range: ${l[@]}"
echo "Batch sizes to probe: ${b[@]}"
echo "Number of epochs for each trial: ${e}"
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
export PYTHONUNBUFFERED=TRUE
python3 hyperparam_optimizer.py --data_folder /work/deodagiu/qml_data/input_ae/ --norm ${n} --aetype ${t} --nevents ${s} --lr ${l[@]}  --batch ${b[@]} --epochs ${e}
export PYTHONUNBUFFERED=FALSE
