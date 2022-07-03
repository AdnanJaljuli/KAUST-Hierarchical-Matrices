#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "utilities.cuh"
// #include "TimingGPU.cuh"

#define FULLSVD 1
#define PRINTRESULTS 0

void SVD(int n, int num_segments, H2Opus_Real* matrix, int maxSegmentSize, H2Opus_Real* h_S, H2Opus_Real* h_U, H2Opus_Real* h_V){
    // printf("SVD\n");
    const int           M = maxSegmentSize;
    const int           N = maxSegmentSize;
    const int           lda = M;
    const int           numMatrices = num_segments;

    // --- Setting the host matrix
    H2Opus_Real *h_A = (H2Opus_Real *)malloc(lda * N * numMatrices * sizeof(double));
    for (unsigned int k = 0; k < numMatrices; k++) {
        for (unsigned int i = 0; i < M; i++) {
            for (unsigned int j = 0; j < N; j++) {
                // h_A[k * M * N + j * M + i] = matrix[(k%num_segments)*N*maxSegmentSize + j*N + (k/num_segments)*maxSegmentSize + i];
                h_A[k * M * N + j * M + i] = matrix[k*N*maxSegmentSize + j*N + i];
            }
        }
    }

    // --- Setting the device matrix and moving the host matrix to the device
    H2Opus_Real *d_A;         gpuErrchk(cudaMalloc(&d_A, M * N * numMatrices * sizeof(H2Opus_Real)));
    gpuErrchk(cudaMemcpy(d_A, h_A, M * N * numMatrices * sizeof(H2Opus_Real), cudaMemcpyHostToDevice));
    free(h_A);
    // --- host side SVD results space
    // H2Opus_Real *h_S = (H2Opus_Real *)malloc(N * numMatrices * sizeof(H2Opus_Real));
    // H2Opus_Real *h_U = NULL;
    // H2Opus_Real *h_V = NULL;
#ifdef FULLSVD
    // h_U = (H2Opus_Real *)malloc(M * M * numMatrices * sizeof(H2Opus_Real));
    // h_V = (H2Opus_Real *)malloc(N * N * numMatrices * sizeof(H2Opus_Real));
#endif

    // --- device side SVD workspace and matrices
    int work_size = 0;

    int *devInfo;        gpuErrchk(cudaMalloc(&devInfo, sizeof(int) * numMatrices));
    H2Opus_Real *d_S;         gpuErrchk(cudaMalloc(&d_S, N * numMatrices * sizeof(H2Opus_Real)));
    H2Opus_Real *d_U = NULL;
    H2Opus_Real *d_V = NULL;
#ifdef FULLSVD
    gpuErrchk(cudaMalloc(&d_U, M * M * numMatrices * sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc(&d_V, N * N * numMatrices * sizeof(H2Opus_Real)));
#endif

    H2Opus_Real *d_work = NULL; /* devie workspace for gesvdj */
    int devInfo_h = 0; /* host copy of error devInfo_h */

    // --- Parameters configuration of Jacobi-based SVD
    const double            tol = 1.e-7;
    const int               maxSweeps = 15;
    cusolverEigMode_t jobz;                                   // --- CUSOLVER_EIG_MODE_VECTOR - Compute eigenvectors; CUSOLVER_EIG_MODE_NOVECTOR - Compute singular values only
#ifdef FULLSVD
    jobz = CUSOLVER_EIG_MODE_VECTOR;
#else
    jobz = CUSOLVER_EIG_MODE_NOVECTOR;
#endif

    const int               econ = 0;                            // --- econ = 1 for economy size 

    // --- Numerical result parameters of gesvdj 
    double                  residual = 0;
    int                     executedSweeps = 0;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle = NULL;
    cusolveSafeCall(cusolverDnCreate(&solver_handle));

    // --- Configuration of gesvdj
    gesvdjInfo_t gesvdj_params = NULL;
    cusolveSafeCall(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    // --- Set the computation tolerance, since the default tolerance is machine precision
    cusolveSafeCall(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

    // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
    cusolveSafeCall(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

    // --- Query the SVD workspace 
    cusolveSafeCall(cusolverDnDgesvdjBatched_bufferSize(
        solver_handle,
        jobz,                                       // --- Compute the singular vectors or not
        M,                                          // --- Number of rows of A, 0 <= M
        N,                                          // --- Number of columns of A, 0 <= N 
        d_A,                                        // --- M x N
        lda,                                        // --- Leading dimension of A
        d_S,                                        // --- Square matrix of size min(M, N) x min(M, N)
        d_U,                                        // --- M x M if econ = 0, M x min(M, N) if econ = 1
        lda,                                        // --- Leading dimension of U, ldu >= max(1, M)
        d_V,                                        // --- N x N if econ = 0, N x min(M,N) if econ = 1
        lda,                                        // --- Leading dimension of V, ldv >= max(1, N)
        &work_size,
        gesvdj_params,
        numMatrices));

    gpuErrchk(cudaMalloc(&d_work, sizeof(H2Opus_Real) * work_size));

    // --- Compute SVD
    // timerGPU.StartCounter();
    cusolveSafeCall(cusolverDnDgesvdjBatched(
        solver_handle,
        jobz,                                       // --- Compute the singular vectors or not
        M,                                          // --- Number of rows of A, 0 <= M
        N,                                          // --- Number of columns of A, 0 <= N 
        d_A,                                        // --- M x N
        lda,                                        // --- Leading dimension of A
        d_S,                                        // --- Square matrix of size min(M, N) x min(M, N)
        d_U,                                        // --- M x M if econ = 0, M x min(M, N) if econ = 1
        lda,                                        // --- Leading dimension of U, ldu >= max(1, M)
        d_V,                                        // --- N x N if econ = 0, N x min(M, N) if econ = 1
        N,                                          // --- Leading dimension of V, ldv >= max(1, N)
        d_work,
        work_size,
        devInfo,
        gesvdj_params,
        numMatrices));

    // printf("Calculation of the singular values only: %f ms\n\n", timerGPU.GetCounter());

    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S, d_S, sizeof(H2Opus_Real) * N * numMatrices, cudaMemcpyDeviceToHost));
#ifdef FULLSVD
    gpuErrchk(cudaMemcpy(h_U, d_U, sizeof(H2Opus_Real) * lda * M * numMatrices, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V, sizeof(H2Opus_Real) * N * N * numMatrices, cudaMemcpyDeviceToHost));
#endif

#if PRINTRESULTS
    printf("SINGULAR VALUES \n");
    printf("_______________ \n");
    for (int k = 0; k < numMatrices; k++)
    {
        for (int p = 0; p < N; p++)
            printf("Matrix nr. %d; SV nr. %d; Value = %lf\n", k, p, h_S[k * N + p]);
        printf("\n");
    }
#if 0 //FULLSVD
    printf("SINGULAR VECTORS U \n");
    printf("__________________ \n");
    for (int k = 0; k < numMatrices; k++)
    {
        for (int q = 0; q < (1 - econ) * M + econ * min(M, N); q++)
            for (int p = 0; p < M; p++)
                printf("Matrix nr. %d; U nr. %d; Value = %lf\n", k, p, h_U[((1 - econ) * M + econ * min(M, N)) * M * k + q * M + p]);
        printf("\n");
    }

    printf("SINGULAR VECTORS V \n");
    printf("__________________ \n");
    for (int k = 0; k < numMatrices; k++)
    {
        for (int q = 0; q < (1 - econ) * N + econ * min(M, N); q++)
            for (int p = 0; p < N; p++)
                printf("Matrix nr. %d; V nr. %d; Value = %lf\n", k, p, h_V[((1 - econ) * N + econ * min(M, N)) * N * k + q * N + p]);
        printf("\n");
    }
#endif
#endif

    if (0 == devInfo_h)
    {
        // printf("gesvdj converges \n");
    }
    else if (0 > devInfo_h)
    {
        printf("%d-th parameter is wrong \n", -devInfo_h);
        exit(1);
    }
    else
    {
        printf("WARNING: devInfo_h = %d : gesvdj does not converge \n", devInfo_h);
    }

    // int offset = maxSegmentSize*maxSegmentSize*num_segments;
    // printf("h_U\n");
    // for(unsigned int i=0; i<M; ++i){
    //     for(unsigned int j=0; j<M;++j){
    //         printf("%lf ", h_U[offset + j*M + i]);
    //     }
    //     printf("\n");
    // }
    // printf("h_S\n");
    // for(unsigned int i=0; i<numMatrices; ++i){
    //         printf("%lf ", h_S[maxSegmentSize*num_segments + i]);
    // }
    // printf("\n");
    // printf("h_V\n");
    // for(unsigned int i=0; i<M; ++i){
    //     for(unsigned int j=0; j<M;++j){
    //         printf("%lf ", h_V[offset + i*M + j]);
    //     }
    //     printf("\n");
    // }

    // H2Opus_Real* tmp = (H2Opus_Real*) malloc(M * M * sizeof(H2Opus_Real));
    // H2Opus_Real* ans = (H2Opus_Real*) malloc(M * M * sizeof(H2Opus_Real));
    // for(int i=0; i<M; ++i){
    //     for(int j=0; j<M; ++j){
    //         ans[j*M + i] = 0;
    //         tmp[j*M + i] = 0;
    //     }
    // }

    // for(int i=0; i<M; ++i){
    //     for(int j=0; j<M; ++j){
    //         tmp[i*M + j] = h_U[offset + i*M + j] * h_S[maxSegmentSize*num_segments + i];
    //     }
    // }

    // printf("tmp\n");
    // for(int i=0; i<M; ++i){
    //     for(int j=0;j<M; ++j){
    //         printf("%lf ", tmp[j*M + i]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for(int i=0; i<M; ++i){
    //     for(int j=0; j<M; ++j){
    //         for(int k=0; k<M; ++k){
    //             // ans[i*M + j] += (tmp[k*M + j] * h_V[i*M + k]);
    //             ans[i*M + j] += (tmp[k*M + j] * h_V[offset + k*M + i]);
    //             // printf("tmp: %lf    V: %lf ", tmp[k*M + j], h_V[i*M + k]);
    //         }
    //         // printf("\n");
    //     }
    //     // printf("\n");
    // }

    // printf("ans\n");
    // for(int i=0; i<M; ++i){
    //     for(int j=0; j<M; ++j){
    //         printf("%lf ", ans[i*M + j]);
    //     }
    //     printf("\n");
    // }

    // --- Free resources
    // printf("ended SVD\n");
    if (d_A) gpuErrchk(cudaFree(d_A));
    if (d_S) gpuErrchk(cudaFree(d_S));
#ifdef FULLSVD
    if (d_U) gpuErrchk(cudaFree(d_U));
    if (d_V) gpuErrchk(cudaFree(d_V));
#endif
    if (devInfo) gpuErrchk(cudaFree(devInfo));
    if (d_work) gpuErrchk(cudaFree(d_work));
    if (solver_handle) cusolveSafeCall(cusolverDnDestroy(solver_handle));
    if (gesvdj_params) cusolveSafeCall(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}