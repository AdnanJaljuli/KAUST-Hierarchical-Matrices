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

void SVD(unsigned int N, unsigned int blockSize, float* matrix, int dim){
    unsigned int nBlocks = (N+blockSize-1)/blockSize;
    const int Nrows = blockSize;
    const int Ncols = blockSize;

    int work_size = 0;
    int *devInfo;           gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    unsigned int k = 3;
    unsigned int numStreams = nBlocks*nBlocks;
    cudaStream_t stream[numStreams];
    for(unsigned int s=0; s<numStreams; ++s){
        cudaStreamCreate(&stream[s]);
    }

    double* Ss = (double*)malloc(min(Nrows, Ncols) * nBlocks*nBlocks * sizeof(double));
    double* Us = (double*)malloc(Nrows * Ncols * nBlocks*nBlocks * sizeof(double));
    double* Vs = (double*)malloc(Nrows * Ncols * nBlocks*nBlocks * sizeof(double));

    // --- host side SVD results space
    double *h_U = (double *)malloc(Nrows * Nrows     * sizeof(double));
    double *h_V = (double *)malloc(Ncols * Ncols     * sizeof(double));
    double *h_S = (double *)malloc(min(Nrows, Ncols) * sizeof(double));

    // --- device side SVD workspace and matrices
    double *d_U;            gpuErrchk(cudaMalloc(&d_U,  Nrows * Nrows     * sizeof(double)));
    double *d_V;            gpuErrchk(cudaMalloc(&d_V,  Ncols * Ncols     * sizeof(double)));
    double *d_S;            gpuErrchk(cudaMalloc(&d_S,  min(Nrows, Ncols) * sizeof(double)));

    for(unsigned int str=0; str<numStreams; ++str){
        double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
        for(int i = 0; i < Nrows; i++){
            for(int j = 0; j < Ncols; j++){
                // h_A[j + i*Nrows] = matrix[(k/nBlocks)*N*blockSize + j*N + (k%nBlocks)*blockSize + i];
                h_A[j*Nrows + i] = matrix[(str%nBlocks)*N*blockSize + j*N + (str/nBlocks)*blockSize + i];
            }
        }

        double *d_A;            gpuErrchk(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
        gpuErrchk(cudaMemcpyAsync(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice, stream[str]));        

        cusolverDnSetStream(solver_handle, stream[str]);
        // --- CUDA SVD initialization
        cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
        double *work;   gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));

        // --- CUDA SVD execution
        cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo));
        int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";

        // --- Moving the results from device to host
        gpuErrchk(cudaMemcpyAsync(h_S, d_S, min(Nrows, Ncols) * sizeof(double), cudaMemcpyDeviceToHost, stream[str]));
        gpuErrchk(cudaMemcpyAsync(h_U, d_U, Nrows * Nrows     * sizeof(double), cudaMemcpyDeviceToHost, stream[str]));
        gpuErrchk(cudaMemcpyAsync(h_V, d_V, Ncols * Ncols     * sizeof(double), cudaMemcpyDeviceToHost, stream[str]));

        for(unsigned int i=0; i<min(Nrows, Ncols); ++i){
            Ss[str*min(Nrows, Ncols) + i] = h_S[i];
        }
        for(unsigned int i=0; i<Nrows*Ncols; ++i){
            Us[str*Nrows*Ncols + i] = h_U[i];
            Vs[str*Nrows*Ncols + i] = h_V[i];
        }
    }

    

    free(h_U);
    free(h_S);
    free(h_V);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaFree(d_U);
    free(Ss);
    free(Us);
    free(Vs);
    cusolverDnDestroy(solver_handle);
}

int main(){
    return 0;
}