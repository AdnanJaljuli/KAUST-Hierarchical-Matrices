
#include "HMatrixVectorMultiplication.cuh"
#include "cublas_v2.h"
#include "cutlassDiagonalXVector.cuh"
#include "cutlassHMatrixXVector.cuh"
#include "HMatrix.cuh"
#include <assert.h>
#include <stdio.h>

// __global__ void checkErrorInDiagXVec(unsigned int size, H2Opus_Real *result, H2Opus_Real *originalResult, H2Opus_Real* error, H2Opus_Real* tmp) {
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     if(i < size) {
//         H2Opus_Real x = originalResult[i];
//         H2Opus_Real y = result[i];
//         printf("x: %lf   y: %lf\n", x, y);
//         atomicAdd(tmp, x*x);
//         atomicAdd(error, (x - y)*(x - y));
//     }
// }

cudaError_t HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {
    
    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0);
    cudaError_t result;
    // multiply diagonal blocks first
    // TODO: replace this with cuBLAS batched gemm
    result = cutlassDiagonalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
    // TODO: add error checking and return error message if result != success

    // #if EXPAND_MATRIX
    // // run cublas on diagonalXVector and compare answers
    // H2Opus_Real *d_tempResultsVectors;
    // cudaMalloc((void**) &d_tempResultsVectors, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    // cublasHandle_t handle;
    // double alpha = 1.0f;
    // double beta = 0.0f;

    // cublasDgemmStridedBatched(handle,
    //     CUBLAS_OP_N, CUBLAS_OP_N,
    //     bucketSize, vectorWidth, bucketSize,
    //     &alpha,
    //     hierarchicalMatrix.diagonalBlocks, bucketSize,
    //     bucketSize*bucketSize,
    //     inpuVectors, numberOfInputPoints,
    //     bucketSize,
    //     &beta,
    //     d_tempResultsVectors, numberOfInputPoints,
    //     bucketSize,
    //     numSegments);

    // H2Opus_Real* d_error;
    // H2Opus_Real* d_tmp;
    // cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    // cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    // cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    // cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));
    
    // unsigned int numThreadsPerBlock = 1024;
    // unsigned int numBlocks = (numberOfInputPoints*vectorWidth + numThreadsPerBlock - 1)/numThreadsPerBlock;
    // printf("matrix size: %d\n", numberOfInputPoints*vectorWidth);
    // cudaDeviceSynchronize();
    // checkErrorInDiagXVec <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints*vectorWidth, resultVectors, d_tempResultsVectors, d_error, d_tmp);
    
    // H2Opus_Real h_error;
    // H2Opus_Real h_tmp;
    // cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    // printf("error in diagXVec: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    // cudaFree(d_tmp);
    // cudaFree(d_error);
    // #endif

    #if 0
    // multiply rest of hierarchical matrix by the vector
    H2Opus_Real *d_bufferVectors;
    cudaMalloc((void**) &d_bufferVectors, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    result = cutlassHierarchicalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix, inpuVectors, d_bufferVectors, resultVectors);
    cudaFree(d_bufferVectors);
    #endif

    return result;
}