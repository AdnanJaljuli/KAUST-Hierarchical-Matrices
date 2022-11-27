
#include "helperFunctions.cuh"
#include <curand.h>

void convertColumnMajorToMorton(unsigned int numSegments, unsigned int maxSegmentSize, uint64_t rankSum, TLR_Matrix matrix, TLR_Matrix &mortonMatrix) {

    cudaMalloc((void**) &mortonMatrix.U, rankSum*maxSegmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.V, rankSum*maxSegmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.blockOffsets, numSegments*numSegments*sizeof(int));
    cudaMalloc((void**) &mortonMatrix.blockRanks, numSegments*numSegments*sizeof(int));
    cudaMalloc((void**) &mortonMatrix.diagonal, static_cast<uint64_t>(numSegments)*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numSegments*numSegments + 1024 - 1)/1024;
    copyCMRanksToMORanks <<< numBlocks, numThreadsPerBlock >>> (numSegments, maxSegmentSize, matrix.blockRanks, mortonMatrix.blockRanks);

    // scan mortonMatrix ranks
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, numSegments*numSegments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, numSegments*numSegments);
    cudaFree(d_temp_storage);

    int* h_matrix_offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    int* h_mortonMatrix_offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(h_matrix_offsets, matrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mortonMatrix_offsets, mortonMatrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    int* h_matrix_ranks = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(h_matrix_ranks, matrix.blockRanks, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i=0; i<numSegments*numSegments; ++i){
        int MOIndex = CMIndextoMOIndex(numSegments, i);
        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (h_matrix_ranks[i]*maxSegmentSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
        assert(h_matrix_ranks[i] >= 0);
        if(h_matrix_ranks[i] > 0){
            cudaMemcpy(&mortonMatrix.U[static_cast<uint64_t>(h_mortonMatrix_offsets[MOIndex])*maxSegmentSize], &matrix.U[static_cast<uint64_t>(h_matrix_offsets[i])*maxSegmentSize], static_cast<uint64_t>(h_matrix_ranks[i])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&mortonMatrix.V[static_cast<uint64_t>(h_mortonMatrix_offsets[MOIndex])*maxSegmentSize], &matrix.V[static_cast<uint64_t>(h_matrix_offsets[i])*maxSegmentSize], static_cast<uint64_t>(h_matrix_ranks[i])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        }
    }

    cudaMemcpy(mortonMatrix.diagonal, matrix.diagonal, numSegments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void copyCMRanksToMORanks(int num_segments, int maxSegmentSize, int* matrixRanks, int* mortonMatrixRanks){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<num_segments*num_segments){
        int MOIndex = CMIndextoMOIndex(num_segments, i);
        mortonMatrixRanks[MOIndex] = matrixRanks[i];
    }
}

void generateRandomVector(unsigned int vectorWidth, unsigned int vectorHeight, H2Opus_Real *vector) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniformDouble(gen, vector, vectorWidth*vectorHeight);
    curandDestroyGenerator(gen);
}

void generateMaxRanks(unsigned int numLevels, unsigned int bucketSize, unsigned int *maxRanks) {
    for(unsigned int i = 0; i < numLevels - 2; ++i) {
        maxRanks[i] = bucketSize*(1 << i);
        if(i > 5) {
            maxRanks[i]/=4;
        }
    }
}