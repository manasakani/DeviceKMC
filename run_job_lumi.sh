#!/bin/bash -l
#SBATCH --job-name=sendhelp # Job name
#SBATCH --output=examplejob.out # Name of stdout output file
#SBATCH --error=examplejob.err  # Name of stderr error file
#SBATCH --partition=standard-g  # or ju-standard-g, partition name
#SBATCH --nodes=1               # Total number of nodes  - 1
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 8 total (1x8) - 8
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank - 8
#SBATCH --cpus-per-task=7       # 7 cpus per task
#SBATCH --time=00:05:00         # Run time (d-hh:mm:ss)
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
# export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK * $SLURM_NTASKS_PER_NODE))
# export OMP_NUM_THREADS=56                                                                     # 7 cpus per task * 8 tasks per node
export OMP_NUM_THREADS=7                                                                     # 7 cpus per task * 8 tasks per node

# cat << EOF > select_gpu
# #!/bin/bash

# export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
# exec \$*
# EOF

# chmod +x ./select_gpu

# 8 GPUs per node
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

# # 7 GPUs per node
# CPU_BIND="mask_cpu:0x00000000000000FE,0x00000000000000FE"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
# CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
# CPU_BIND="${CPU_BIND},0xFE000000,0xFE000000"

# # 6 GPUs per node
# CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
# CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
# CPU_BIND="${CPU_BIND},0xFE000000,0xFE000000"

# # 5 GPUs per node
# CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
# CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
# CPU_BIND="${CPU_BIND},0xFE0000,0xFE0000"

# # 4 GPUs per node
# CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
# CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
# CPU_BIND="${CPU_BIND},0xFE00,0xFE00"

# # 3 GPUs per node
# CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
# CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
# CPU_BIND="${CPU_BIND},0xFE,0xFE"

# # 2 GPUs per node
# CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
# CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"

# # 1 GPU per node
# CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
# CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"

# cd ./structures/single_devices/timing_20nm/ 
cd ./structures/crossbars/20nm_3x3/ 
# srun --cpu-bind=${CPU_BIND} ../../../bin/runKMC parameters.txt
srun ../../../bin/runKMC parameters.txt

# srun --cpu-bind=${CPU_BIND}  rocgdb ../../../bin/runKMC parameters.txt
# srun --cpu-bind=${CPU_BIND}  rocgdb ../../../bin/runKMC parameters.txt
# srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 ../../../bin/runKMC parameters.txt

# module load cray-python/3.10.10                                                           
# check cores per task (srun help) - 8 cores per task
# right now it uses 1 core
# srun gpu_check -l
# !! double check linkage of GPU Transfer Libraries for MPI-aware CUDA with: $ ldd ./bin/runKMC | grep libmpi !!
# Debug: set all to blocking - export AMD_SERIALIZE_KERNEL=3
