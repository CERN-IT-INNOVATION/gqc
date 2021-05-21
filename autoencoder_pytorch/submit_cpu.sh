#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --mem=3000M
#SBATCH -o ./logs/ae_train_%j.log

usage() { echo "Usage: $0 [-t <training_file_path>] [-v <validation_file_path>] [-l <learning_rate>] [-b <batch_size>] [-e <number_of_epochs>] [-f <file_flag>]" 1>&2; exit 1; }

while getopts ":t:v:l:b:e:s:f:" o; do
    case "${o}" in
    t)
        t=${OPTARG}
            ;;
    v)
        v=${OPTARG}
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

if [ -z "${t}" ] || [ -z "${v}" ] || [ -z "${l}" ] || [ -z "${b}" ] || [ -z "${e}" ] || [ -z "${f}" ]; then
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Input training file: ${t}"
echo "Input validation file: ${v}"
echo "Learning rate: ${l}"
echo "Batch size: ${b}"
echo "Number of Epochs: ${e}"
echo "File flag: ${f}"
echo " "
echo "--------------------------------------- "

source /work/deodagiu/miniconda3/bin/activate qml_project
python3 main.py --training_file /work/deodagiu/qml_data/${t} --validation_file /work/deodagiu/qml_data/${v} --lr ${l} --batch ${b} --epochs ${e} --file_flag ${f}
