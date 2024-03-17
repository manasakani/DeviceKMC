#!/bin/bash -l
#SBATCH --job-name=CG # Job name
#SBATCH --output=examplejob.out # Name of stdout output file
#SBATCH --error=examplejob.err  # Name of stderr error file
#SBATCH --partition=standard-g  # or ju-standard-g, partition name
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 8 total (1x8) - 8
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank - 8
# SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000929

# check cores per task (srun help) - 8 cores per task #SBATCH --ncpu-per-task=8
# right now it uses 1 core

# The carefully assembled compile and runtime environment (DO NOT CHANGE ORDER)...:
# module load cray-python/3.10.10
# module restore kmc_env                                                                      # Restore module environment used for kmc:
# module load craype-x86-trento                                                               # CPU used in the LUMI-G partition
# export OMP_NUM_THEADS=7  
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"    # GPU Transfer Library - allows hipcc to behave like {CC}
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)                                 # GPU Transfer Library - allows hipcc to behave like {CC}
export MPICH_GPU_SUPPORT_ENABLED=1                                                          # Enable GPU-aware MPI  
export MPICH_MAX_THREAD_SAFETY=multiple                                                     # Enable multiple OMP threads to communicate using MPI

# !! double check linkage of GPU Transfer Libraries for MPI-aware CUDA with: $ ldd ./bin/runKMC | grep libmpi !!
# Debug: set all to blocking - export AMD_SERIALIZE_KERNEL=3

export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_FORCE_GPU_AWARE_MPI=1

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

#cd ./structures/single_devices/timing_10nm/ 
# srun rocprof --hip-trace ./main #profile
# srun --cpu-bind=${CPU_BIND} ./select_gpu ./main
srun --cpu-bind=${CPU_BIND} ./select_gpu rocprof --hip-trace ./main #profile

# cat << EOF > select_gpu
# #!/bin/bash
# export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
# exec \$*
# EOF
# chmod +x ./select_gpu
# CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
# rm -rf ./select_gpu

# cd ./structures/single_devices/timing_2.5nm/ ; srun  ../../../bin/runKMC parameters.txt

