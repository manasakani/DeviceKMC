#!/bin/bash --login 
#SBATCH --account=s1212
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --time=10:00:00
#SBATCH --job-name=kmc-4node


export OMP_NUM_THREADS=12

#module swap PrgEnv-cray PrgEnv-gnu/6.0.10
# module load intel-oneapi/2022.1.0
# module load daint-gpu/21.09
# module load cudatoolkit/21.3_11.2
# module swap gcc gcc/9.3.0

#cd ./tests/1-potential/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
#cd ./tests/2-globaltemp/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
# cd ./tests/3-localtemp/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
#cd ./tests/prod_2xstructure/ ; srun -n $SLURM_NTASKS ../../bin/runKMC parameters.txt
cd ./structures/crossbars/40nm_3x3/ ; srun ../../../bin/runKMC parameters.txt
# cd ./structures/single_devices/timing_10nm/ ; srun  ../../../bin/runKMC parameters.txt