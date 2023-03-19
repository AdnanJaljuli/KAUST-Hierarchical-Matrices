# Overview
This is an imlpementation of generating hierarchical matrices on GPUs. This code is written using CUDA and C/C++.

# Folders
- src: contains the source code of the project
- test: you can disregard this folder

# Instructions

To build KAUST_GEMM, please follow these instructions:

1.  Go to  [lr-kblas-gpu](https://github.com/AdnanJaljuli/lr-kblas-gpu.git) and follow the installation instructions.

2.  Edit file Makefile to:
    - Provide path for third party libraries (_KBLAS_ROOT_, _MAGMA_ROOT_, _CUDA_ROOT_, _OPENBLAS_ROOT_, _CUTLASS_ROOT_).
    - Specify CUDA architecture to compile for (GPU_ARCH).
    - Remove DEBUG_FLAGS if wanted.

    or

    - Provide equivalent environment variables.

3.  Build KAUST_GEMM
    - make

    or
    
    
    - with counters: make USE_COUNTERS=1

    and/or
    
    - with error checking (won't run efficiently): make EXPAND_MATRIX=1
