#!/bin/sh
#SBATCH --job-name=vqc_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=10:00:00
#SBATCH -o ./logs/vqc_gpu_%j.out

usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>]"\
               "[-p <model_path>] [-f <output_folder>] [-q <nb_of_qubits>]"\
               "[-v <vform_repeats>] [-o <optimizer>] [-e <epochs>] [-b <batch_size>]"\
               "[-h <hybrid> {0 or 1}] [-c <class_weight>] [-a <ntrain>] [-l <nvalid>]"\
               "[-g <learning_rate>] [-r <run_type>] [-k <backend_name>]"\
               "[-d <diff_method>] [-i <ideal_device>]" 1>&2; exit 1; }

while getopts ":n:s:p:f:q:v:o:e:b:h:c:a:l:g:r:k:d:i:" x; do
    case "${x}" in
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
    q)
        q=${OPTARG}
        ;;
    v)
        v=${OPTARG}
        ;;
    o)
        o=${OPTARG}
        ;;
    e)
        e=${OPTARG}
        ;;
    b)
        b=${OPTARG}
        ;;
    h)
        h=${OPTARG}
        ;;
    c)
        c=${OPTARG}
        ;;
    a)
        a=${OPTARG}
        ;;
    l) 
        l=${OPTARG}
        ;;
    g)
        g=${OPTARG}
        ;;
    r)
        r=${OPTARG}
        ;;
    k)
        k=${OPTARG}
        ;;
    d) 
        d=${OPTARG}
        ;;
    i)
        i=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${p}" ] || [ -z "${f}" ] || [ -z "${q}" ] || \
   [ -z "${v}" ] || [ -z "${o}" ] || [ -z "${e}" ] || [ -z "${b}" ] || [ -z "${h}" ] || \
   [ -z "${c}" ] || [ -z "${a}" ] || [ -z "${l}" ] || [ -z "${r}" ] || [ -z "${k}" ] || \
   [ -z "${g}" ] || [ -z "${d}" ] || [ -z "${i}" ] ; then
    usage
fi

if [ $h -eq 0 ] ; then
        h=""
elif [ $h -eq 1 ] ; then
        h="--hybrid" # we use argparse action=store_value
else 
    usage
fi

echo "--------------------------------------- "
echo "The hyperparameters of this run are:"
echo " "
echo "Normalization of the input file: ${n}"
echo "Number of events in the input file: ${s}"
echo "Hybrid VQC: ${h}"
echo "Differentiation method of the quantum circuit: ${d}"
echo "Learning rate: ${g}"
echo "Batch size: ${b}"
echo "Number of Epochs: ${e}"
echo "Output folder name: ${f}"
echo "--------------------------------------- "

source /work/deodagiu/miniconda/bin/activate ae_qml_pnl
export PYTHONUNBUFFERED=TRUE
srun vqc_train --data_folder /work/deodagiu/data/ae_input --norm ${n} --nevents ${s} \
               --model_path ${p} --output_folder ${f} --nqubits ${q} --vform_repeats ${v} \
               --optimiser ${o} --epochs ${e} --learning_rate ${g} --batch_size ${b} ${h} \
               --class_weight ${c} --ntrain ${a} --nvalid ${l} --run_type ${r} \
               --backend_name ${k} --diff_method ${d} --ideal_dev ${i}
export PYTHONUNBUFFERED=FALSE

mv ./logs/vqc_gpu_${SLURM_JOBID}.out ./trained_vqcs/${f}/