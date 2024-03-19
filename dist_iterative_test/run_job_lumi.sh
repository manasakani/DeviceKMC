#!/bin/bash -l
#SBATCH --job-name=dog_roc         # Job name
#SBATCH --partition=standard-g  # or ju-standard-g, partition name
#SBATCH --nodes=1               # Total number of nodes  - 1
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 8 total (1x8) - 8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank - 8
#SBATCH --cpus-per-task=7       # 7 cpus per tasl
#SBTCCH --mem-per-GPU=60G       # Memory per gpu
#SBATCH --time=00:03:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000929

# The carefully assembled compile and runtime environment (DO NOT CHANGE ORDER)...:
# module restore kmc_env                                                                      # module environment used for kmc
# module load craype-x86-trento                                                               # CPU used in the LUMI-G partition
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"      # GPU Transfer Library - allows hipcc to behave like {CC}
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)                                   # GPU Transfer Library - allows hipcc to behave like {CC}

# export MPICH_MAX_THREAD_SAFETY=multiple                                                       # Enable multiple OMP threads to communicate using MPI

export MPICH_GPU_SUPPORT_ENABLED=1                                                            # Enable GPU-aware MPI  
export GMX_FORCE_GPU_AWARE_MPI=1
export GMX_ENABLE_DIRECT_GPU_COMM=1

export MPICH_OFI_NIC_POLICY=GPU

export HCC_AMDGPU_TARGET=gfx90a

export OMP_NUM_THREADS=7
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# cat << EOF > select_gpu

# #!/bin/bash

# export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
# exec \$*
# EOF

# chmod +x ./select_gpu

if [ $SLURM_NTASKS_PER_NODE -eq 8 ]; then
    # 8 GPUs per node
    CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
    CPU_BIND="${CPU_BIND},fe0000,fe000000"
    CPU_BIND="${CPU_BIND},fe,fe00"
    CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"
elif [ $SLURM_NTASKS_PER_NODE -eq 7 ]; then
    # 7 GPUs per node
    CPU_BIND="mask_cpu:0x00000000000000FE,0x00000000000000FE"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE000000,0xFE000000"
elif [ $SLURM_NTASKS_PER_NODE -eq 6 ]; then
    # 6 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE000000,0xFE000000"
elif [ $SLURM_NTASKS_PER_NODE -eq 5 ]; then
    # 5 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE0000,0xFE0000"
elif [ $SLURM_NTASKS_PER_NODE -eq 4 ]; then
    # 4 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE00,0xFE00"
elif [ $SLURM_NTASKS_PER_NODE -eq 3 ]; then
    # 3 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE,0xFE"
elif [ $SLURM_NTASKS_PER_NODE -eq 2 ]; then
    # 2 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
elif [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
    # 1 GPU per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
else
    echo "Unsupported number of GPUs per node!"
    exit 1
fi

# srun ./wrapper.sh --hip-trace --hsa-trace  ./main
srun --cpu-bind=${CPU_BIND} ./select_gpu ./main
# srun ./select_gpu ./main

