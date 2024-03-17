#!/bin/bash -l
#SBATCH --job-name=dog_roc         # Job name
#SBATCH --output=examplejob.out # Name of stdout output file
#SBATCH --error=examplejob.err  # Name of stderr error file
#SBATCH --partition=standard-g  # or ju-standard-g, partition name
#SBATCH --nodes=1               # Total number of nodes  - 1
#SBATCH --ntasks-per-node=2     # 8 MPI ranks per node, 8 total (1x8) - 8
#SBATCH --gpus-per-node=2       # Allocate one gpu per MPI rank - 8
#SBATCH --cpus-per-task=7       # 7 cpus per tasl
#SBTCCH --mem-per-GPU=60G       # Memory per gpu
#SBATCH --time=00:02:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000929

# The carefully assembled compile and runtime environment (DO NOT CHANGE ORDER)...:
# module restore kmc_env                                                                      # module environment used for kmc
# module load craype-x86-trento                                                               # CPU used in the LUMI-G partition
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"      # GPU Transfer Library - allows hipcc to behave like {CC}
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)                                   # GPU Transfer Library - allows hipcc to behave like {CC}
export MPICH_MAX_THREAD_SAFETY=multiple                                                       # Enable multiple OMP threads to communicate using MPI
export MPICH_GPU_SUPPORT_ENABLED=1                                                            # Enable GPU-aware MPI  
export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_FORCE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK))

# srun rocprof --hip-trace ./main #profile
srun ./main

