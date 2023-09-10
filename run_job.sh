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

export OMP_NUM_THREADS=4
module load daint-gpu

cd ./tests/1-potential/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
