# Overview
This is an imlpementation of generating hierarchical matrices on GPUs. This code is written using CUDA and C/C++.

# Folders
- src: Contains the source code of the project
- test: you can disregard this folder

# Instructions



Installation
============

To build KAUST_GEMM, please follow these instructions:

1.  Go to  [lr-kblas-gpu](https://github.com/AdnanJaljuli/lr-kblas-gpu.git) and follow the installation instructions

2.  Edit file make.inc to:
    - Enable / disable KBLAS sub modules (_SUPPORT_BLAS2_, _SUPPORT_BLAS3_, _SUPPORT_BATCH_TR_, _SUPPORT_SVD_, _SUPPORT_TLR_).
    - Enable / disable usage of third party libraries (_USE_MKL_, _USE_MAGMA_) for performance comparisons.
    - Provide path for third party libraries if required (_CUB_DIR_, _MAGMA_ROOT_).
    - Specify CUDA architecture to compile for (_CUDA_ARCH_).

    or

    - Provide equivalent environment variables.

3.  Build KBLAS

        make
