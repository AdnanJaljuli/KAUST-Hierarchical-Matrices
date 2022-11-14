
#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__

#include <utility>
#include <cstdint> 
#include "cublas_v2.h"
#include "kblas.h"
#include "TLRMatrix.h"
#include "helperKernels.cuh"
#include <cub/cub.cuh>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

__global__ void generateDataset_kernel(int numberOfInputPoints, int dimensionOfInputPoints, H2Opus_Real* pointCloud) {
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < numberOfInputPoints*dimensionOfInputPoints) {
        unsigned int seed = i;
        curandState s;
        curand_init(seed, 0, 0, &s);
        pointCloud[i] = curand_uniform(&s);
    }
}

static void generateDataset(int numberOfInputPoints, int dimensionOfInputPoints, H2Opus_Real* d_pointCloud) {
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numberOfInputPoints*dimensionOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateDataset_kernel <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, d_pointCloud);
    cudaDeviceSynchronize();
}

static __device__ __host__ int upperPowerOfTwo(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static bool isPowerOfTwo (int v) {
    return v && (!(v&(v - 1)));
}

static std::pair<int, int> getMaxSegmentSize(int n, int bucket_size) {
    int it = 0;
    while(n > bucket_size) {
        n = (n + 1)/2;
        ++it;
    }
    std::pair<int, int> p;
    p.first = n;
    p.second = it;
    return p;
}

static void convertColumnMajorToMorton(uint64_t numSegments, uint64_t maxSegmentSize, uint64_t rankSum, TLR_Matrix matrix, TLR_Matrix &mortonMatrix) {

    cudaMalloc((void**) &mortonMatrix.U, rankSum*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.V, rankSum*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.blockOffsets, numSegments*numSegments*(uint64_t)sizeof(int));
    cudaMalloc((void**) &mortonMatrix.blockRanks, numSegments*numSegments*(uint64_t)sizeof(int));
    cudaMalloc((void**) &mortonMatrix.diagonal, numSegments*maxSegmentSize*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numSegments*numSegments + 1024 - 1)/1024;
    copyCMRanksToMORanks <<< numBlocks, numThreadsPerBlock >>> (numSegments, maxSegmentSize, matrix.blockRanks, mortonMatrix.blockRanks);
    cudaDeviceSynchronize();

    // scan mortonMatrix ranks
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, numSegments*numSegments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, numSegments*numSegments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    int* h_matrix_offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    int* h_mortonMatrix_offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(h_matrix_offsets, matrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mortonMatrix_offsets, mortonMatrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    int* h_matrix_ranks = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(h_matrix_ranks, matrix.blockRanks, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i=0; i<numSegments*numSegments; ++i){
        int MOIndex = CMIndextoMOIndex_h(numSegments, i);
        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (h_matrix_ranks[i]*maxSegmentSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
        assert(h_matrix_ranks[i] >= 0);
        if(h_matrix_ranks[i] > 0){
            copyTilestoMO <<< numBlocks, numThreadsPerBlock >>> (h_matrix_ranks[i]*maxSegmentSize, mortonMatrix.U, matrix.U, h_mortonMatrix_offsets[MOIndex]*maxSegmentSize, h_matrix_offsets[i]*maxSegmentSize);
            copyTilestoMO <<< numBlocks, numThreadsPerBlock >>> (h_matrix_ranks[i]*maxSegmentSize, mortonMatrix.V, matrix.V, h_mortonMatrix_offsets[MOIndex]*maxSegmentSize, h_matrix_offsets[i]*maxSegmentSize);
        }
    }

    cudaMemcpy(mortonMatrix.diagonal, matrix.diagonal, numSegments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

void printMatrix(int numberOfInputPoints, int numSegments, int segmentSize, TLR_Matrix matrix, int level, int rankSum) {
    // n=512, numLevels=5, level=2, batchSize=4, numTilesInBatch=4
    int* ranks = (int*)malloc(numSegments*numSegments*sizeof(int));
    int* offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(ranks, matrix.blockRanks, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(offsets, matrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    H2Opus_Real* U = (H2Opus_Real*)malloc(rankSum*segmentSize*sizeof(H2Opus_Real));
    H2Opus_Real* V = (H2Opus_Real*)malloc(rankSum*segmentSize*sizeof(H2Opus_Real));
    cudaMemcpy(U, matrix.U, rankSum*segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, matrix.V, rankSum*segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    char fileName[100] = "batchedMatrix.txt";
    FILE *outputFile = fopen(fileName, "w");
    int batchSize = 2;
    int unitSize = 4;
    // int tilesToPrint[64] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239};
    int tilesToPrint[32] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    fprintf(outputFile, "%d %d %d\n", unitSize, segmentSize, batchSize);
    int cnt = 0;
    for(unsigned int i = 0; i < numSegments*numSegments; ++i) {
        if(i == tilesToPrint[cnt]) {
            ++cnt;
            fprintf(outputFile, "%d\n", ranks[i]);
            for(unsigned int j = 0; j < ranks[i]*segmentSize; ++j) {
                fprintf(outputFile, "%lf ", U[offsets[i]*segmentSize + j]);
            }
            fprintf(outputFile, "\n");
            for(unsigned int j = 0; j < ranks[i]*segmentSize; ++j) {
                fprintf(outputFile, "%lf ", V[offsets[i]*segmentSize + j]);
            }
            fprintf(outputFile, "\n");
        }
    }
    fclose(outputFile);
}

static void printDenseMatrix(H2Opus_Real* d_denseMatrix, int size) {
    H2Opus_Real* denseMatrix = (H2Opus_Real*)malloc(size*sizeof(H2Opus_Real));
    cudaMemcpy(denseMatrix, d_denseMatrix, size*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    char fileName[100] = "denseMatrix.txt";
    FILE *outputFile = fopen(fileName, "w");
    for(unsigned int row = 0; row < 256; ++row) {
        for(unsigned int col = 256; col < 512; ++col) {
            fprintf(outputFile, "%lf ", denseMatrix[col*512 + row]);
        }
        fprintf(outputFile, "\n");
    }
}

#endif