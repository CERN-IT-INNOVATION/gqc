#!/bin/sh
#SBATCH --job-name=vqc_train
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5000M
#SBATCH --time=7-00:00:00
#SBATCH -o ./logs/vqc_cpu_%j.out

usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>] [-p <model_path>] [-l <loss_name>] [-q <nb_of_qubits>] [-e <epochs>] [-b <batch_size>] [-f <output_folder>]" 1>&2; exit 1; }

while getopts ":n:s:p:l:q:e:b:f:o:" o; do
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
    q)
        q=${OPTARG}
        ;;
    e)
        e=${OPTARG}
        ;;
    b)
        b=${OPTARG}
        ;;
    f)
        f=${OPTARG}
        ;;
    o)
        o=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${p}" ] || [ -z "${o}" ] || [ -z "${q}" ] || [ -z "${e}" ] || [ -z "${b}" ] || [ -z "${f}" ]; then
    usage
fi

source /work/deodagiu/miniconda/bin/activate ae_qml
export PYTHONUNBUFFERED=TRUE
python vqc_train --data_folder /work/deodagiu/data/ae_input --norm ${n} --nevents ${s} --model_path ${p} --nqubits ${q} --epochs ${e} --batch_size ${b} --output_folder ${f} --optimiser ${o}
export PYTHONUNBUFFERED=FALSE

mv ./logs/vqc_cpu_${SLURM_JOBID}.out ./trained_vqcs/${f}/
