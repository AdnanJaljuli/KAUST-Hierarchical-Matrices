#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "utilities.cuh"
// #include "TimingGPU.cuh"

#define FULLSVD 1
#define PRINTRESULTS 0

void SVD(int n, int num_segments, H2Opus_Real* matrix, int nBlocks, int NRows, int NCols, int maxSegmentSize){

    const int           M = NRows;
    const int           N = NCols;
    const int           lda = M;
    const int           numMatrices = num_segments*num_segments;

    // --- Setting the host matrix
    cuComplex *h_A = (cuComplex *)malloc(lda * N * numMatrices * sizeof(double));
    for (unsigned int k = 0; k < numMatrices; k++)
        for (unsigned int i = 0; i < M; i++)
        {
            for (unsigned int j = 0; j < N; j++)
            {
                h_A[k * M * N + j * M + i] = make_float2(matrix[(k%num_segments)*N*maxSegmentSize + j*N + (k/num_segments)*maxSegmentSize + i], matrix[(k%num_segments)*N*maxSegmentSize + j*N + (k/num_segments)*maxSegmentSize + i]);
                h_A[k * M * N + j * M + i] = make_float2((1. / (k + 1)) * (i + j * j) * (i + j), (1. / (k + 1)) * (i + j * j) * (i + j));
            }
        }

    // --- Setting the device matrix and moving the host matrix to the device
    cuComplex *d_A;         gpuErrchk(cudaMalloc(&d_A, M * N * numMatrices * sizeof(cuComplex)));
    gpuErrchk(cudaMemcpy(d_A, h_A, M * N * numMatrices * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // --- host side SVD results space
    float *h_S = (float *)malloc(N * numMatrices * sizeof(float));
    cuComplex *h_U = NULL;
    cuComplex *h_V = NULL;
#ifdef FULLSVD
    h_U = (cuComplex *)malloc(M * M * numMatrices * sizeof(cuComplex));
    h_V = (cuComplex *)malloc(N * N * numMatrices * sizeof(cuComplex));
#endif

    // --- device side SVD workspace and matrices
    int work_size = 0;

    int *devInfo;        gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));
    float *d_S;         gpuErrchk(cudaMalloc(&d_S, N * numMatrices * sizeof(float)));
    cuComplex *d_U = NULL;
    cuComplex *d_V = NULL;
#ifdef FULLSVD
    gpuErrchk(cudaMalloc(&d_U, M * M * numMatrices * sizeof(cuComplex)));
    gpuErrchk(cudaMalloc(&d_V, N * N * numMatrices * sizeof(cuComplex)));
#endif

    cuComplex *d_work = NULL; /* devie workspace for gesvdj */
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
    cusolveSafeCall(cusolverDnCgesvdjBatched_bufferSize(
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

    gpuErrchk(cudaMalloc(&d_work, sizeof(cuComplex) * work_size));

    // --- Compute SVD
    // timerGPU.StartCounter();
    cusolveSafeCall(cusolverDnCgesvdjBatched(
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
    gpuErrchk(cudaMemcpy(h_S, d_S, sizeof(float) * N * numMatrices, cudaMemcpyDeviceToHost));
#ifdef FULLSVD
    gpuErrchk(cudaMemcpy(h_U, d_U, sizeof(cuComplex) * lda * M * numMatrices, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V, sizeof(cuComplex) * N * N * numMatrices, cudaMemcpyDeviceToHost));
#endif

#ifdef PRINTRESULTS
    printf("SINGULAR VALUES \n");
    printf("_______________ \n");
    for (int k = 0; k < numMatrices; k++)
    {
        for (int p = 0; p < N; p++)
            printf("Matrix nr. %d; SV nr. %d; Value = %f\n", k, p, h_S[k * N + p]);
        printf("\n");
    }
#if 0 //FULLSVD
    printf("SINGULAR VECTORS U \n");
    printf("__________________ \n");
    for (int k = 0; k < numMatrices; k++)
    {
        for (int q = 0; q < (1 - econ) * M + econ * min(M, N); q++)
            for (int p = 0; p < M; p++)
                printf("Matrix nr. %d; U nr. %d; Value = %f\n", k, p, h_U[((1 - econ) * M + econ * min(M, N)) * M * k + q * M + p]);
        printf("\n");
    }

    printf("SINGULAR VECTORS V \n");
    printf("__________________ \n");
    for (int k = 0; k < numMatrices; k++)
    {
        for (int q = 0; q < (1 - econ) * N + econ * min(M, N); q++)
            for (int p = 0; p < N; p++)
                printf("Matrix nr. %d; V nr. %d; Value = %f\n", k, p, h_V[((1 - econ) * N + econ * min(M, N)) * N * k + q * N + p]);
        printf("\n");
    }
#endif
#endif

    if (0 == devInfo_h)
    {
        printf("gesvdj converges \n");
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

    // --- Free resources
    // if (d_A) gpuErrchk(cudaFree(d_A));
    // if (d_S) gpuErrchk(cudaFree(d_S));
// #ifdef FULLSVD
//     if (d_U) gpuErrchk(cudaFree(d_U));
//     if (d_V) gpuErrchk(cudaFree(d_V));
// #endif
//     if (devInfo) gpuErrchk(cudaFree(devInfo));
//     if (d_work) gpuErrchk(cudaFree(d_work));
//     if (solver_handle) cusolveSafeCall(cusolverDnDestroy(solver_handle));
//     if (gesvdj_params) cusolveSafeCall(cusolverDnDestroyGesvdjInfo(gesvdj_params));

//     gpuErrchk(cudaDeviceReset());
}