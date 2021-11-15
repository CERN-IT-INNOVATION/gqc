#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4000
#SBATCH -o ./trained_aes/logs/ae_cpu_%j.out

# Define the folder where the input data is found.
# Change so it is valid for your own machine.
DATA_FOLDER=/work/deodagiu/data/ae_input/

usage() { echo "Usage: $0 [-n <normalization_name>] [-t <autencoder_type>] [-s <number_of_events>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>] [-f <file_flag>]" 1>&2; exit 1; }

while getopts ":n:t:s:q:w:l:b:e:f:" o; do
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

export PYTHONUNBUFFERED=TRUE
pipenv run python ae_train --data_folder $DATA_FOLDER --norm ${n} --aetype ${t} --nevents ${s} --lr ${l} --batch ${b} --epochs ${e} --outdir ${f}
export PYTHONUNBUFFERED=FALSE
