#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<math.h>

#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include "utilities.cuh"

void SVD(unsigned int N, unsigned int blockSize, float* matrix){

    // --- gesvd only supports Nrows >= Ncols
    // --- column major memory ordering
    unsigned int nBlocks = (N+blockSize-1)/blockSize;
    const int Nrows = blockSize;
    const int Ncols = blockSize;

    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;           gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    unsigned int k = 3;
    // (k/nBlocks)*N*blockSize + threadIdx.x*N + (k%nBlocks)*blockSize + threadIdx.y

    // --- Setting the host, Nrows x Ncols matrix
    double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
    for(int i = 0; i < Nrows; i++){
        for(int j = 0; j < Ncols; j++){
            h_A[j + i*Nrows] = matrix[(k/nBlocks)*N*blockSize + j*N + (k%nBlocks)*blockSize + i];
        }
    }
    

    // for(int j = 0; j < Nrows; j++)
    //     for(int i = 0; i < Ncols; i++)
    //         h_A[j + i*Nrows] = (i + j*j) * sqrt((double)(i + j));

    // --- Setting the device matrix and moving the host matrix to the device
    double *d_A;            gpuErrchk(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

    // --- host side SVD results space
    double *h_U = (double *)malloc(Nrows * Nrows     * sizeof(double));
    double *h_V = (double *)malloc(Ncols * Ncols     * sizeof(double));
    double *h_S = (double *)malloc(min(Nrows, Ncols) * sizeof(double));

    // --- device side SVD workspace and matrices
    double *d_U;            gpuErrchk(cudaMalloc(&d_U,  Nrows * Nrows     * sizeof(double)));
    double *d_V;            gpuErrchk(cudaMalloc(&d_V,  Ncols * Ncols     * sizeof(double)));
    double *d_S;            gpuErrchk(cudaMalloc(&d_S,  min(Nrows, Ncols) * sizeof(double)));

    // --- CUDA SVD initialization
    cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
    double *work;   gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));

    // --- CUDA SVD execution
    cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo));
    int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";

    // --- Moving the results from device to host
    gpuErrchk(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V, Ncols * Ncols     * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "Singular values\n";
    for(int i = 0; i < min(Nrows, Ncols); i++) 
        std::cout << "d_S["<<i<<"] = " << std::setprecision(15) << h_S[i] << std::endl;

    std::cout << "\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n";
    for(int j = 0; j < Nrows; j++) {
        printf("\n");
        for(int i = 0; i < Nrows; i++)
            printf("U[%i,%i]=%f\n",i,j,h_U[j*Nrows + i]);
    }

    std::cout << "\nRight singular vectors - For y = A * x, the columns of V span the space of x\n";
    for(int i = 0; i < Ncols; i++) {
        printf("\n");
        for(int j = 0; j < Ncols; j++)
            printf("V[%i,%i]=%f\n",i,j,h_V[j*Ncols + i]);
    }

    cusolverDnDestroy(solver_handle);


}