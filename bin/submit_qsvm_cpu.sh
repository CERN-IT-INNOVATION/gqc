#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20000
#SBATCH -o ./qsvm_logs/ae_train_cpu_%j.out

mkdir -p ./qsvm_logs
usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>] [-p <model_path>] [-f <output_file>]" 1>&2; exit 1; }

while getopts ":n:s:p:f:" o; do
    case "${o}" in
    n)
        n=${OPTARG}
        ;;
    s)
        s=${OPTARG}
        ;;
    p)
        p=${OPTARG}
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

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${p}" ] || [ -z "${f}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Normalization of the input file: ${n}"
echo "Number of events in the input file: ${s}"
echo "Model path: ${p}"
echo "File flag: ${f}"
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate legacy_qiskit
export PYTHONUNBUFFERED=TRUE
python3 legacy_launchQSVM.py --data_folder /work/deodagiu/qml_data/input_ae/ --norm ${n} --nevents ${s} --model_path ${p} --output_file ${f}
export PYTHONUNBUFFERED=FALSE
