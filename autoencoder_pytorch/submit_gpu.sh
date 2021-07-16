#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/ae_train_gpu_%j.log

mkdir -p ./logs
usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>] [-f <file_flag>]" 1>&2; exit 1; }

while getopts ":n:s:l:b:e:f:" o; do
    case "${o}" in
    n)
        n=${OPTARG}
        ;;
    s)
        s=${OPTARG}
        ;;
    l)
        l=${OPTARG}
        ;;
    b)
        b=${OPTARG}
        ;;
    e)
        e=${OPTARG}
        ;;
    f)
        f=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ] || [ -z "${f}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Normalization of the input file: ${n}"
echo "Number of events in the input file: ${s}"
echo "Learning rate: ${l}"
echo "Batch size: ${b}"
echo "Number of Epochs: ${e}"
echo "File flag: ${f}"
echo " "
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
export PYTHONUNBUFFERED=TRUE
python3 train.py --data_folder /work/deodagiu/qml_data/input_ae/ --norm ${n} --nevents ${s} --lr ${l} --batch ${b} --epochs ${e} --file_flag ${f}
export PYTHONUNBUFFERED=FALSE
