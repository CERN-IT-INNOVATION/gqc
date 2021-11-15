#!/bin/sh
#SBATCH --job-name=ae_optuna
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --time=7-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1
#SBATCH --output=trained_models/logs/ae_optim_gpu_%j.log

# Folder where the data is located.
# Change for your own configuration.
DATA_FOLDER=/work/deodagiu/data/ae_input/

usage() { echo "Usage: $0 [-n <normalization_name>] [-t <autencoder_type>] [-s <number_of_events>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>] [-w <weight_learning_bool>]" 1>&2; exit 1; }

while getopts ":n:t:s:l:b:e:w:" o; do
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
    w)
        w=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${n}" ] || [ -z "${t}" ] || [ -z "${s}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ] || [ -z "${w}"]; then
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
echo "Weight optimisation: ${w}"
echo "--------------------------------------- "

export PYTHONUNBUFFERED=TRUE
pipenv run python hyperparam_optimizer.py --data_folder $DATA_FOLDER --norm ${n} --aetype ${t} --nevents ${s} --lr ${l[@]}  --batch ${b[@]} --epochs ${e} --woptim ${w}
export PYTHONUNBUFFERED=FALSE
