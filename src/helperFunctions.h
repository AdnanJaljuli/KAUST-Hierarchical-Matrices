#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <utility>
#include <cstdint> 
#include "cublas_v2.h"
#include "helperKernels.cuh"
#include <cub/cub.cuh>
#define numTimers 13

void printCountersInFile(float* times){
    char filename[100] = "results/times.csv";
    FILE *output_file = fopen(filename, "a");
    for(unsigned int i = 0; i<numTimers; ++i){
        fprintf(output_file,"%f, ",times[i]);
    }
    fprintf(output_file, "\n");
    fclose(output_file);
}

__device__ __host__ int upper_power_of_two(int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

bool isPowerOfTwo (int x) {
    return x && (!(x&(x-1)));
}

std::pair<int, int> getMaxSegmentSize(int n, int bucket_size){
    int it=0;
    while(n > bucket_size){
        n = (n + 1)/2;
        ++it;
    }
    std::pair<int, int> p;
    p.first = n;
    p.second = it;
    return p;
}

void printSigmas(H2Opus_Real* S, uint64_t num_segments, uint64_t maxSegmentSize, int bucket_size, int n, int segment){
    FILE *fp;
    fp = fopen("results/sigma_values.csv", "a");// "w" means that we are going to write on this file
    if(segment == 0){
        fprintf(fp, "bucket size: %d, n: %d, num segments: %d,\n", bucket_size, n, num_segments);
    }

    for(unsigned int i=0; i<num_segments; ++i){
        for(unsigned int j=0; j<maxSegmentSize; ++j){
            fprintf(fp, "%lf, ", S[i*maxSegmentSize + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp); //Don't forget to close the file when finished
}

void printKs(int* K, uint64_t num_segments, uint64_t maxSegmentSize, int bucket_size, int n){
    FILE *fp;
    fp = fopen("results/K_values.csv", "a");// "w" means that we are going to write on this file

    for(unsigned int i=0; i<num_segments; ++i){
        fprintf(fp, "%d, ", K[i]);
    }
    fprintf(fp, "\n");
    fclose(fp); //Don't forget to close the file when finished
}

void ColumnMajorToMorton(int num_segments, int maxSegmentSize, int k_sum, TLR_Matrix matrix, TLR_Matrix mortonMatrix){

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (num_segments*num_segments + 1024 - 1)/1024;
    copyCMRanksToMORanks<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, matrix.blockRanks, mortonMatrix.blockRanks);
    cudaDeviceSynchronize();

    // printK<<<1, 1>>>(matrix.blockRanks, num_segments*num_segments);
    // printK<<<1, 1>>>(mortonMatrix.blockRanks, num_segments*num_segments);

    // scan mortonMatrix ranks
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, num_segments*num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    // copyCMTilesToMOTiles<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, matrix, mortonMatrix);
    // cudaDeviceSynchronize();

    int* h_matrix_offsets = (int*)malloc(num_segments*num_segments*sizeof(int));
    int* h_mortonMatrix_offsets = (int*)malloc(num_segments*num_segments*sizeof(int));
    cudaMemcpy(h_matrix_offsets, matrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mortonMatrix_offsets, mortonMatrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost);

    int* h_matrix_ranks = (int*)malloc(num_segments*num_segments*sizeof(int));
    cudaMemcpy(h_matrix_ranks, matrix.blockRanks, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i=0; i<num_segments*num_segments; ++i){
        int MOIndex = IndextoMOIndex_h(num_segments, i);
        unsigned int x = i%num_segments;
        unsigned int y = i/num_segments;

        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (h_matrix_ranks[i] + numThreadsPerBlock - 1)/numThreadsPerBlock;
        printf("ranks[i]: %d\n", h_matrix_ranks[i]);
        assert(h_matrix_ranks[i] >= 0);
        if(h_matrix_ranks[i] > 0){
            copyTiles<<<numBlocks, numThreadsPerBlock>>>(h_matrix_ranks[i]*maxSegmentSize, mortonMatrix.U, matrix.U, h_mortonMatrix_offsets[MOIndex]*maxSegmentSize, h_matrix_offsets[i]*maxSegmentSize);
            copyTiles<<<numBlocks, numThreadsPerBlock>>>(h_matrix_ranks[i]*maxSegmentSize, mortonMatrix.V, matrix.V, h_mortonMatrix_offsets[MOIndex]*maxSegmentSize, h_matrix_offsets[i]*maxSegmentSize);
        }
    }

    cudaMemcpy(mortonMatrix.diagonal, matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
}

// void MortonToColumnMajor(int num_segments, int maxSegmentSize, int k_sum, TLR_Matrix matrix, TLR_Matrix mortonMatrix){

//     unsigned int numThreadsPerBlock = 1024;
//     unsigned int numBlocks = (num_segments*num_segments + 1024 - 1)/1024;
//     copyMORanksToCMRanks<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, matrix.blockRanks, mortonMatrix.blockRanks);
//     cudaDeviceSynchronize();

//     // scan mortonMatrix ranks
//     void* d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
//     cudaDeviceSynchronize();
//     cudaFree(d_temp_storage);

//     copyMOTilestoCMTiles<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, matrix, mortonMatrix);
//     cudaDeviceSynchronize();

//     cudaMemcpy(mortonMatrix.diagonal, matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize,cudaMemcpyDeviceToDevice);
// }

#endif