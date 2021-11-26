#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=0-07:00
#SBATCH --output=./logs/ae_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/deodagiu/data/ae_input/

usage() { echo "Usage: $0 [-n <normalization_name>] [-t <autencoder_type>] [-s <number_of_events>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>] [-f <file_flag>]" 1>&2; exit 1; }

while getopts ":n:t:s:l:b:e:f:" o; do
    case "${o}" in
    n)
        n=${OPTARG}
        ;;
    t)
	t=${OPTARG}
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

if [ -z "${n}" ] || [ -z "${t}" ] || [ -z "${s}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ] || [ -z "${f}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Normalization of the input file: ${n}"
echo "Autoencoder type: ${t}"
echo "Number of events in the input file: ${s}"
echo "Learning rate: ${l}"
echo "Batch size: ${b}"
echo "Number of Epochs: ${e}"
echo "File flag: ${f}"
echo "--------------------------------------- "

source /work/deodagiu/miniconda/bin/activate ae_qml
export PYTHONUNBUFFERED=TRUE
python ae_train --data_folder $DATA_FOLDER --norm ${n} --aetype ${t} --nevents ${s} --lr ${l} --batch ${b} --epochs ${e} --outdir ${f}
export PYTHONUNBUFFERED=FALSE
