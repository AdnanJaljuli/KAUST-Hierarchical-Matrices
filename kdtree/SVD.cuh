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

void SVD(unsigned int N, unsigned int blockSize, H2Opus_Real* matrix, H2Opus_Real* Ss, H2Opus_Real* Us, H2Opus_Real Vs){
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

    // H2Opus_Real* Ss = (H2Opus_Real*)malloc(min(Nrows, Ncols) * nBlocks*nBlocks * sizeof(H2Opus_Real));
    // H2Opus_Real* Us = (H2Opus_Real*)malloc(Nrows * Ncols * nBlocks*nBlocks * sizeof(H2Opus_Real));
    // H2Opus_Real* Vs = (H2Opus_Real*)malloc(Nrows * Ncols * nBlocks*nBlocks * sizeof(H2Opus_Real));

    // --- host side SVD results space
    H2Opus_Real *h_U = (H2Opus_Real *)malloc(Nrows * Nrows     * sizeof(H2Opus_Real));
    H2Opus_Real *h_V = (H2Opus_Real *)malloc(Ncols * Ncols     * sizeof(H2Opus_Real));
    H2Opus_Real *h_S = (H2Opus_Real *)malloc(min(Nrows, Ncols) * sizeof(H2Opus_Real));

    // --- device side SVD workspace and matrices
    H2Opus_Real *d_U;            gpuErrchk(cudaMalloc(&d_U,  Nrows * Nrows     * sizeof(H2Opus_Real)));
    H2Opus_Real *d_V;            gpuErrchk(cudaMalloc(&d_V,  Ncols * Ncols     * sizeof(H2Opus_Real)));
    H2Opus_Real *d_S;            gpuErrchk(cudaMalloc(&d_S,  min(Nrows, Ncols) * sizeof(H2Opus_Real)));

    for(unsigned int str=0; str<numStreams; ++str){
        H2Opus_Real *h_A = (H2Opus_Real *)malloc(Nrows * Ncols * sizeof(H2Opus_Real));
        for(int i = 0; i < Nrows; i++){
            for(int j = 0; j < Ncols; j++){
                // h_A[j + i*Nrows] = matrix[(k/nBlocks)*N*blockSize + j*N + (k%nBlocks)*blockSize + i];
                h_A[j*Nrows + i] = matrix[(str%nBlocks)*N*blockSize + j*N + (str/nBlocks)*blockSize + i];
            }
        }

        H2Opus_Real *d_A;            gpuErrchk(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(H2Opus_Real)));
        gpuErrchk(cudaMemcpyAsync(d_A, h_A, Nrows * Ncols * sizeof(H2Opus_Real), cudaMemcpyHostToDevice, stream[str]));        

        cusolverDnSetStream(solver_handle, stream[str]);
        // --- CUDA SVD initialization
        cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
        H2Opus_Real *work;   gpuErrchk(cudaMalloc(&work, work_size * sizeof(H2Opus_Real)));

        // --- CUDA SVD execution
        cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo));
        int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";

        // --- Moving the results from device to host
        gpuErrchk(cudaMemcpyAsync(h_S, d_S, min(Nrows, Ncols) * sizeof(H2Opus_Real), cudaMemcpyDeviceToHost, stream[str]));
        gpuErrchk(cudaMemcpyAsync(h_U, d_U, Nrows * Nrows     * sizeof(H2Opus_Real), cudaMemcpyDeviceToHost, stream[str]));
        gpuErrchk(cudaMemcpyAsync(h_V, d_V, Ncols * Ncols     * sizeof(H2Opus_Real), cudaMemcpyDeviceToHost, stream[str]));

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
    cusolverDnDestroy(solver_handle);
}
