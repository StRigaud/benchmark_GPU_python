#!/bin/bash
#SBATCH --job-name=benchmark_gpu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=g
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=clip-g1-1,clip-g2-1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00

CONTAINER=$HOME/containers/pixi.sif

# CUDA Forward Compatibility libs (see: docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
# The cluster's R535 driver (535.183.01, CUDA 12.2) is too old to JIT-load PTX
# emitted by pyclesperanto_cuda/cupy's bundled CUDA 12.6 nvrtc. This directory
# holds a newer libcuda.so (extracted from the official cuda-compat-12-6 .deb)
# that still talks to the same R535 kernel module, bridging the gap.
CUDA_COMPAT_DIR="$HOME/containers/cuda-compat-12-6"

# Bind the host's OpenCL ICD registry so the NVIDIA OpenCL platform is discoverable inside the container (--nv alone does not do this for ocl).
APPTAINER_BINDS="$PWD:$PWD"
if [ -d /etc/OpenCL/vendors ]; then
    APPTAINER_BINDS="$APPTAINER_BINDS,/etc/OpenCL/vendors:/etc/OpenCL/vendors"
fi
APPTAINER_BINDS_CUDA="$APPTAINER_BINDS,$CUDA_COMPAT_DIR:$CUDA_COMPAT_DIR:ro"

# Do NOT load a host CUDA module: the pixi env bundles its own CUDA runtime/nvrtc
# for cupy and pyclesperanto_cuda. Leaking a host module's LD_LIBRARY_PATH (which
# includes its compile-time stub libcuda.so) into the container shadows the real
# GPU driver library bound in by --nv, making the driver look like an old version
# and breaking PTX JIT loading. --cleanenv guarantees no such leakage happens.
module unload cuda 2>/dev/null || true
unset CUDA_HOME CUDA_PATH CUDA_ROOT EBROOTCUDA EBVERSIONCUDA EBDEVELCUDA LD_LIBRARY_PATH

# The cuda-compat package only replaces libcuda.so (+ debugger/JIT libs), NOT
# libnvidia-opencl.so, which stays at the host's real R535 version. If both the
# CUDA and OpenCL benchmarks run in the SAME process, cupy/pyclesperanto_cuda
# load the newer forward-compat libcuda.so first, and the OpenCL ICD (built
# against the old R535 libcuda ABI) then reuses that already-loaded, wrong
# version instead of its matching sibling — an unsupported, version-skewed
# pairing that "mostly works" on small buffers but hangs/crashes on larger
# ones (observed killing the whole job at the 2048MB OpenCL benchmark).
# Fix: run OpenCL benchmarks in a separate process that never loads the
# compat libcuda.so, so it only ever touches the real, matched driver libs.
apptainer exec --cleanenv --nv \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "LD_LIBRARY_PATH=${CUDA_COMPAT_DIR}" \
    --bind "$APPTAINER_BINDS_CUDA" --pwd $PWD "$CONTAINER" pixi run benchmark-cuda

apptainer exec --cleanenv --nv \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --bind "$APPTAINER_BINDS" --pwd $PWD "$CONTAINER" pixi run benchmark-opencl