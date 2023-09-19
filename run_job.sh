#!/bin/bash --login 
#SBATCH --account=hck
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00
#SBATCH --job-name=test
#SBATCH --output=run.out
#SBATCH --error=run.err

export OMP_NUM_THREADS=4

#module swap PrgEnv-cray PrgEnv-gnu/6.0.10
module load intel-oneapi/2022.1.0
module load daint-gpu/21.09
module load cudatoolkit/21.3_11.2
module swap gcc gcc/9.3.0

#cd ./tests/1-potential/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
#cd ./tests/2-globaltemp/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
#cd ./tests/3-localtemp/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
cd ./tests/prod_2xstructure/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt