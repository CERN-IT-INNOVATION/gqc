#!/bin/sh
#SBATCH --job-name=ae_train
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=3000M
#SBATCH -o ./logs/qsvm_cpu_%j.out

usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>] [-p <model_path>] [-d <qsvm_name>] [-c <qsvm_constant_c>] [-f <output_folder>]" 1>&2; exit 1; }

while getopts ":n:s:p:d:c:f:" o; do
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
    d)
        d=${OPTARG}
        ;;
    c)
        c=${OPTARG}
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

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${p}" ] || [ -z "${d}" ] || [ -z "${b}" ] || [ -z "${r}" ]; then
    usage
fi

source /work/vabelis/miniconda3/bin/activate ae_qml
export PYTHONUNBUFFERED=TRUE
qsvm_test --data_folder /work/vabelis/data/ae_input --norm ${n} --nevents ${s} --model_path ${p} 
--display_name ${d} --backend_name ${b} --run_type ${r}
export PYTHONUNBUFFERED=FALSE
