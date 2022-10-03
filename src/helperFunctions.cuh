
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
        int MOIndex = IndextoMOIndex_h(numSegments, i);
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

#endif
