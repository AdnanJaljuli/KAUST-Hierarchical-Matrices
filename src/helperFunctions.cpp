#include "helperFunctions.h"

void generateDataset_h(int n, int dim, H2Opus_Real* &d_dataset){
    gpuErrchk(cudaMalloc((void**) &d_dataset, n*dim*(uint64_t)sizeof(H2Opus_Real)));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateDataset<<<numBlocks, numThreadsPerBlock>>> (n, dim, d_dataset);
    cudaDeviceSynchronize();
}

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

void ConvertColumnMajorToMorton(uint64_t num_segments, uint64_t maxSegmentSize, uint64_t k_sum, TLR_Matrix matrix, TLR_Matrix &mortonMatrix){

    cudaMalloc((void**) &mortonMatrix.U, k_sum*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.V, k_sum*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.blockOffsets, num_segments*num_segments*(uint64_t)sizeof(int));
    cudaMalloc((void**) &mortonMatrix.blockRanks, num_segments*num_segments*(uint64_t)sizeof(int));
    cudaMalloc((void**) &mortonMatrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (num_segments*num_segments + 1024 - 1)/1024;
    copyCMRanksToMORanks<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, matrix.blockRanks, mortonMatrix.blockRanks);
    cudaDeviceSynchronize();

    // scan mortonMatrix ranks
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, num_segments*num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    int* h_matrix_offsets = (int*)malloc(num_segments*num_segments*sizeof(int));
    int* h_mortonMatrix_offsets = (int*)malloc(num_segments*num_segments*sizeof(int));
    cudaMemcpy(h_matrix_offsets, matrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mortonMatrix_offsets, mortonMatrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost);

    int* h_matrix_ranks = (int*)malloc(num_segments*num_segments*sizeof(int));
    cudaMemcpy(h_matrix_ranks, matrix.blockRanks, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i=0; i<num_segments*num_segments; ++i){
        int MOIndex = IndextoMOIndex_h(num_segments, i);

        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (h_matrix_ranks[i]*maxSegmentSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
        assert(h_matrix_ranks[i] >= 0);
        if(h_matrix_ranks[i] > 0){
            copyTilestoMO<<<numBlocks, numThreadsPerBlock>>>(h_matrix_ranks[i]*maxSegmentSize, mortonMatrix.U, matrix.U, h_mortonMatrix_offsets[MOIndex]*maxSegmentSize, h_matrix_offsets[i]*maxSegmentSize);
            copyTilestoMO<<<numBlocks, numThreadsPerBlock>>>(h_matrix_ranks[i]*maxSegmentSize, mortonMatrix.V, matrix.V, h_mortonMatrix_offsets[MOIndex]*maxSegmentSize, h_matrix_offsets[i]*maxSegmentSize);
        }
    }
    
    cudaMemcpy(mortonMatrix.diagonal, matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

void checkErrorInLRMatrix(uint64_t num_segments, uint64_t max_segment_size, TLR_Matrix matrix, H2Opus_Real* d_denseMatrix){
    H2Opus_Real* d_expandedMatrix;
    cudaMalloc((void**) &d_expandedMatrix, num_segments*max_segment_size*num_segments*max_segment_size*sizeof(H2Opus_Real));

    dim3 mm_numBlocks(num_segments, num_segments);
    dim3 mm_numThreadsPerBlock(32, 32);
    expandMatrix<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_expandedMatrix, matrix);

    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));

    errorInMatrix<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_denseMatrix, d_expandedMatrix, d_error, d_tmp);

    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_expandedMatrix);
}