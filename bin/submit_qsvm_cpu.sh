#!/bin/sh
#SBATCH --job-name=qsvm_train
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4000M
#SBATCH --time=7-00:00:00
#SBATCH -o ./logs/qsvm_cpu_%j.out

usage() { echo "Usage: $0 [-n <normalization_name>] [-s <number_of_events>]"\
                "[-p <model_path>] [-c <qsvm_constant_c>] [-f <output_folder>]"\
                "[-b <backend_name>] [-r <run_type>] [-a <ntrain>]"\
                "[-v <nvalid>] [-t <ntest>]" 1>&2; exit 1; }

while getopts ":n:s:p:c:f:b:r:a:v:t:" o; do
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
    c)
        c=${OPTARG}
        ;;
    f)
        f=${OPTARG}
        ;;
    b)
        b=${OPTARG}
        ;;
    r)
        r=${OPTARG}
        ;;
    a)
        a=${OPTARG}
        ;;
    v)
        v=${OPTARG}
        ;;
    t)
        t=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${n}" ] || [ -z "${s}" ] || [ -z "${p}" ] || [ -z "${c}" ] || [ -z "${f}" ] || [ -z "${r}" ] || [ -z "${a}" ] || [ -z ${v} ] || [ -z ${t} ]; then
    usage
fi

source /work/vabelis/miniconda3/bin/activate ae_qml
export PYTHONUNBUFFERED=TRUE
./qsvm_launch --data_folder /work/vabelis/data/ae_input --norm ${n} \
              --nevents ${s} --model_path ${p} --c_param ${c} \
              --output_folder ${f} --backend_name ${b} --run_type ${r} \
              --ntrain ${a} --nvalid ${v} --ntest ${t}
export PYTHONUNBUFFERED=FALSE
# Move the stdout file to the corresponding trained model folder for more
# efficient job testing. Also remove the ibm_ or ibmq_ prefix.
b=`sed -e s'/ibmq\?_'//g <<< ${b}`
mv ./logs/qsvm_cpu_${SLURM_JOB_ID}.out models/${f}_c\=${c}_${r}_${b}
