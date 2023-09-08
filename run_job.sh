#!/bin/bash --login 
#SBATCH --account=
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00
#SBATCH --job-name=test
#SBATCH --output=run.out
#SBATCH --error=run.err

export OMP_NUM_THREADS=1
export CRAY_CUDA_MPS=1
module load daint-gpu
#module swap PrgEnv-cray PrgEnv-gnu/6.0.10
module load intel-oneapi/2022.1.0

srun -n $SLURM_NTASKS ./bin/runKMC parameters.txt
