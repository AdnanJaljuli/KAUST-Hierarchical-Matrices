#include "tlr_example.h"
#include "TLR_Matrix.cuh"
#include "helperFunctions.h"
#include "helperKernels.cuh"
#include "config.h"
#include "kdtreeConstruction.cuh"
#include "createLRMatrix.cuh"

#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <typeinfo>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

#define EXPAND_MATRIX 1
#define BLOCK_SIZE 32
using namespace std;

int main(int argc, char *argv[]){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice name: %s\n\n", prop.name);

    cudaEvent_t startCode, stopCode;
    cudaEventCreate(&startCode);
    cudaEventCreate(&stopCode);
    cudaEventRecord(startCode);

    Config config = parseArgs(argc, argv);
    printf("n: %d\n", config.n);
    printf("bucket size: %d\n", config.bucket_size);
    printf("epsilon: %f\n", config.tol);
    printf("dim: %d\n", config.dim);
    float tolerance = config.tol;

    float* timer_arr = (float*)malloc(numTimers*sizeof(float));
    timer_arr[0] = (float)config.n;
    timer_arr[1] = (float)config.bucket_size;
    timer_arr[2] = (float)config.dim;
    timer_arr[3] = (float)config.tol;

    cudaError_t cudaErr;
    H2Opus_Real *d_dataset;
    gpuErrchk(cudaMalloc((void**) &d_dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real)));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (config.n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateDataset<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, d_dataset);
    cudaDeviceSynchronize();

    uint64_t num_segments = 1;
    uint64_t max_num_segments = (config.n+config.bucket_size-1)/config.bucket_size;

    printf("max num segments: %d\n", max_num_segments);

    int  *d_values_in;
    int  *d_offsets_sort;
    gpuErrchk(cudaMalloc((void**) &d_offsets_sort, (max_num_segments + 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_values_in, config.n*sizeof(int)));

    cudaEvent_t startKDtree, stopKDtree;
    cudaEventCreate(&startKDtree);
    cudaEventCreate(&stopKDtree);
    cudaEventRecord(startKDtree);
    createKDTree(config.n, config.dim, config.bucket_size, num_segments, config.div_method, d_values_in, d_offsets_sort, d_dataset, max_num_segments);
    cudaEventRecord(stopKDtree);
    cudaEventSynchronize(stopKDtree);
    cudaEventElapsedTime(&timer_arr[4], startKDtree, stopKDtree);
    cudaEventDestroy(startKDtree);
    cudaEventDestroy(stopKDtree);

    uint64_t max_segment_size = config.bucket_size;
    printf("max segment size: %lu\n", max_segment_size);
    printf("num segments: %lu\n", num_segments);
    TLR_Matrix matrix;
    H2Opus_Real* d_denseMatrix;

    const int ARA_R = 10;
    int max_rows = max_segment_size;
    int max_cols = max_segment_size;
    int max_rank = max_cols;

    uint64_t k_sum = createLRMatrix(config.n, num_segments, max_segment_size, config.bucket_size, config.dim, matrix, d_denseMatrix, d_values_in, d_offsets_sort, d_dataset, tolerance, ARA_R, max_rows, max_cols, max_rank);

    // TODO: check error in matrix against epxanded matrix
    #if EXPAND_MATRIX
    H2Opus_Real* d_expandedCMMatrix;
    cudaMalloc((void**) &d_expandedCMMatrix, num_segments*max_segment_size*num_segments*max_segment_size*sizeof(H2Opus_Real));
    dim3 mm_numBlocks(num_segments, num_segments);
    dim3 mm_numThreadsPerBlock(32, 32);
    expandCMMatrix<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_expandedCMMatrix, matrix);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    H2Opus_Real* d_error;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    H2Opus_Real* h_error = (H2Opus_Real*)malloc(sizeof(H2Opus_Real));
    *h_error = 0;
    cudaMemcpy(d_error, h_error, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    H2Opus_Real* h_tmp = (H2Opus_Real*)malloc(sizeof(H2Opus_Real));
    *h_tmp = 0;
    cudaMemcpy(d_tmp, h_tmp, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());
    errorInCMMatrix<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_denseMatrix, d_expandedCMMatrix, d_error, d_tmp);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error in column major ordered matrix: %lf\n", sqrt(*h_error)/sqrt(*h_tmp));
    free(h_tmp);
    free(h_error);
    cudaFree(d_tmp);
    cudaFree(d_error);
    #endif
    gpuErrchk(cudaPeekAtLastError());
    // TODO: make createLRMatrix a function

    TLR_Matrix mortonMatrix;
    cudaMalloc((void**) &mortonMatrix.U, k_sum*max_segment_size*(uint64_t)sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.V, k_sum*max_segment_size*(uint64_t)sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.blockOffsets, num_segments*num_segments*(uint64_t)sizeof(int));
    cudaMalloc((void**) &mortonMatrix.blockRanks, num_segments*num_segments*(uint64_t)sizeof(int));
    cudaMalloc((void**) &mortonMatrix.diagonal, num_segments*max_segment_size*max_segment_size*(uint64_t)sizeof(H2Opus_Real));
    ColumnMajorToMorton(num_segments, max_segment_size, k_sum, matrix, mortonMatrix);
    gpuErrchk(cudaPeekAtLastError());

    printK<<<1, 1>>>(mortonMatrix.blockRanks, num_segments*num_segments);
    printK<<<1, 1>>>(mortonMatrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();

    #if EXPAND_MATRIX
    H2Opus_Real* d_expandedMOMatrix;
    cudaMalloc((void**) &d_expandedMOMatrix, num_segments*max_segment_size*num_segments*max_segment_size*sizeof(H2Opus_Real));
    expandMOMatrix<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_expandedMOMatrix, mortonMatrix);
    gpuErrchk(cudaPeekAtLastError());

    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    h_error = (H2Opus_Real*)malloc(sizeof(H2Opus_Real));
    *h_error = 0;
    cudaMemcpy(d_error, h_error, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    h_tmp = (H2Opus_Real*)malloc(sizeof(H2Opus_Real));
    *h_tmp = 0;
    cudaMemcpy(d_tmp, h_tmp, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());

    errorInMOMatrix<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_denseMatrix, d_expandedMOMatrix, d_error, d_tmp);
    cudaMemcpy(h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error in morton ordered matrix: %lf\n", sqrt(*h_error)/sqrt(*h_tmp));
    free(h_tmp);
    free(h_error);
    cudaFree(d_tmp);
    cudaFree(d_error);

    compareMOwithCM<<<mm_numBlocks, mm_numThreadsPerBlock>>>(num_segments, max_segment_size, d_expandedCMMatrix, d_expandedMOMatrix);
    // cudaFree(d_expandedCMMatrix);
    // cudaFree(d_expandedMOMatrix);
    gpuErrchk(cudaPeekAtLastError());
    #endif

    const int num_levels = __builtin_ctz(config.n) - __builtin_ctz(config.bucket_size);
    printf("num_levels: %d\n", num_levels);
    int** HMatrixRanks = (int**)malloc(num_levels*sizeof(int*));
    int** HMatrixAvailableTiles = (int**)malloc(num_levels*sizeof(int*));
    int num_available_tiles = num_segments*(num_segments-1);

    int *d_rows_batch, *d_cols_batch, *d_ranks;
    int *d_lda_batch, *d_ldb_batch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_A_ptrs, **d_B_ptrs;

    cudaMalloc((void**) &HMatrixRanks[0], num_available_tiles*sizeof(int));
    cudaMalloc((void**) &HMatrixAvailableTiles[0], num_available_tiles*sizeof(int));

    // TODO: parallelize
    fillFirstLevelAvailableArrays<<<1, 1>>>(num_segments, HMatrixAvailableTiles[0], HMatrixRanks[0], mortonMatrix.blockRanks);
    unsigned int tile_size = config.bucket_size;
    bool stop = false;

    #if 1
    for(unsigned int level = 0; level < (num_levels - 1); ++level){
        // TODO: set cudaMalloc and cudaFrees to outside the loop
        // TODO: use cudaMemSet to initialize num_ops to 0
        int* d_num_ops;
        cudaMalloc((void**) &d_num_ops, sizeof(int));
        // TODO: no need to dynamically allocate h_num_ops. just pass the reference.
        int* h_num_ops = (int*)malloc(sizeof(int));
        *h_num_ops = 0;
        cudaMemcpy(d_num_ops, h_num_ops, sizeof(int), cudaMemcpyHostToDevice);
        numThreadsPerBlock = 1024;
        numBlocks = (num_available_tiles + numThreadsPerBlock - 1)/numThreadsPerBlock;

        // TODO: instead of using atmoicAdds, let each thread write to a bit vector and then do a reduce
        calcNumOps<<<numBlocks, numThreadsPerBlock>>> (num_available_tiles, d_num_ops, HMatrixAvailableTiles[level]);        
        cudaMemcpy(h_num_ops, d_num_ops, sizeof(int), cudaMemcpyDeviceToHost);
        int num_ops = *h_num_ops;
        printf("level: %d   num ops: %d\n", level, num_ops);

        free(h_num_ops);
        cudaFree(d_num_ops);

        // TODO: rename available tiles to existing tiles
        int* d_activeTiles;
        int* d_activeRanks;
        cudaMalloc((void**) &d_activeTiles, 4*num_ops*sizeof(int));
        cudaMalloc((void**) &d_activeRanks, 4*num_ops*sizeof(int));

        // TODO: parallelize
        fillActiveTiles<<<1, 1>>>(num_available_tiles, d_activeTiles, HMatrixAvailableTiles[level], d_activeRanks, HMatrixRanks[0]);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        printK<<<1, 1>>>(d_activeTiles, num_ops*4);

        max_rows <<= 1;
        max_cols <<= 1;
        max_rank <<= 1;
        // tolerance *= 2;
        printf("max rows: %d\n", max_rows);
        printf("tolerance: %f\n", tolerance);

        // TODO: find a tight upper limit and malloc and free before and after the loop
        gpuErrchk(cudaMalloc((void**) &d_ranks, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_rows_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_cols_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_lda_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_ldb_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_A_ptrs, num_ops*sizeof(H2Opus_Real*)));
        gpuErrchk(cudaMalloc((void**) &d_B_ptrs, num_ops*sizeof(H2Opus_Real*)));
        gpuErrchk(cudaMalloc((void**) &d_A, num_ops*max_rows*max_rank*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_B, num_ops*max_rows*max_rank*sizeof(H2Opus_Real)));

        // TODO: parallelize
        fillLRARAArrays<<<1, 1>>>(num_ops, max_rows, max_cols, d_rows_batch, d_cols_batch, d_lda_batch, d_ldb_batch);

        generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols, num_ops, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols, num_ops, 0);
        gpuErrchk(cudaPeekAtLastError());

        kblasHandle_t kblas_handle_2;
        kblasRandState_t rand_state_2;
        kblasCreate(&kblas_handle_2);

        kblasInitRandState(kblas_handle_2, &rand_state_2, 1<<15, 0);
        gpuErrchk(cudaPeekAtLastError());

        kblasEnableMagma(kblas_handle_2);
        kblas_gesvj_batch_wsquery<H2Opus_Real>(kblas_handle_2, max_rows, max_cols, num_ops);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle_2, config.bucket_size, num_ops);
        kblasAllocateWorkspace(kblas_handle_2);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        // TODO: optimize max_cols. max_cols shouldn't be equal to max_rows, instead, its values should depend on the ranks of the tiles
        int lr_ARA_return = lr_kblas_ara_batch(kblas_handle_2, d_rows_batch, d_cols_batch, mortonMatrix.U, mortonMatrix.V, d_activeRanks, mortonMatrix.blockOffsets, d_activeTiles,
            d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks,
            tolerance, max_rows, max_cols, max_rank, 32, ARA_R, rand_state_2, 0, num_ops
        );
        cudaDeviceSynchronize();
        assert(lr_ARA_return == 1);
        gpuErrchk(cudaPeekAtLastError());

        #if EXPAND_MATRIX
        cudaDeviceSynchronize();
        H2Opus_Real* expandedMatrix;
        cudaMalloc((void**) &expandedMatrix, num_ops*max_rows*max_cols*sizeof(H2Opus_Real));
        dim3 hm_numBlocks(2, 2*num_ops);
        dim3 hm_numThreadsPerBlock(32, 32);
        // expandHMatrixLevel<<<hm_numBlocks, hm_numThreadsPerBlock>>>(num_ops, 64, 64, d_A_ptrs, d_B_ptrs, d_ranks, expandedMatrix);
        expandHMatrixLevel<<<hm_numBlocks, hm_numThreadsPerBlock>>>(num_ops, 64, 64, d_A, d_B, d_ranks, expandedMatrix);

        H2Opus_Real* d_error;
        cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
        H2Opus_Real* h_error = (H2Opus_Real*)malloc(sizeof(H2Opus_Real));
        *h_error = 0;
        cudaMemcpy(d_error, h_error, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        H2Opus_Real* d_tmp;
        cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
        H2Opus_Real* h_tmp = (H2Opus_Real*)malloc(sizeof(H2Opus_Real));
        *h_tmp = 0;
        cudaMemcpy(d_tmp, h_tmp, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        errorInHMatrix<<<hm_numBlocks, hm_numThreadsPerBlock>>>(num_segments, max_segment_size, num_ops, max_rows, max_cols, expandedMatrix, d_denseMatrix, d_activeTiles, d_error, d_tmp);
        cudaDeviceSynchronize();
        cudaMemcpy(h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        printf("h matrix error: %lf\n", sqrt(*h_error)/sqrt(*h_tmp));

        // printK<<<1, 1>>>(d_ranks, num_ops);
        #endif
        break;

        // TODO: optimize the bit vector: use an array of longs instead.
        int* d_old_bit_vector;
        int* d_new_bit_vector;
        int* d_old_bit_vector_scan;
        int* d_new_bit_vector_scan;
        gpuErrchk(cudaMalloc((void**) &d_old_bit_vector, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_new_bit_vector, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_old_bit_vector_scan, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_new_bit_vector_scan, num_ops*sizeof(int)));

        numThreadsPerBlock = 1024;
        numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_ops, tile_size, d_ranks, d_activeRanks, d_new_bit_vector, d_old_bit_vector);
        cudaDeviceSynchronize();
        tile_size <<= 1;

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_old_bit_vector, d_old_bit_vector_scan, num_ops);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_old_bit_vector, d_old_bit_vector_scan, num_ops);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_new_bit_vector, d_new_bit_vector_scan, num_ops);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_new_bit_vector, d_new_bit_vector_scan, num_ops);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* d_newLevelCount;
        int* newLevelCount = (int*)malloc(sizeof(int));
        cudaMalloc((void**) &d_newLevelCount, sizeof(int));
        getNewLevelCount<<<1, 1>>>(num_ops, d_new_bit_vector, d_new_bit_vector_scan, d_newLevelCount);
        cudaMemcpy(newLevelCount, d_newLevelCount, sizeof(int), cudaMemcpyDeviceToHost);
        num_available_tiles = *newLevelCount;
        printf("new level count %d\n", num_available_tiles);

        if(*newLevelCount == 0) {
            stop = true;
        }
        else {
            gpuErrchk(cudaMalloc((void**) &HMatrixRanks[level + 1], *newLevelCount*sizeof(int)));
            gpuErrchk(cudaMalloc((void**) &HMatrixAvailableTiles[level + 1], *newLevelCount*sizeof(int)));

            numThreadsPerBlock = 1024;
            numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
            fillNewLevel<<<numBlocks, numThreadsPerBlock>>>(num_ops, d_new_bit_vector, d_new_bit_vector_scan, d_ranks, HMatrixRanks[level + 1], d_activeTiles, HMatrixAvailableTiles[level + 1]);
            copyTilesToNewLevel<<<numBlocks, numThreadsPerBlock>>>(num_ops, d_new_bit_vector, mortonMatrix, d_A, d_B, d_ranks, d_activeTiles, max_rows, max_cols);
            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            // TODO: clean previous ranks and active tiles arrays
        }
        kblasDestroy(&kblas_handle_2);
        kblasDestroyRandState(rand_state_2);
        free(newLevelCount);
        cudaFree(d_newLevelCount);

        cudaFree(d_ranks);
        cudaFree(d_rows_batch);
        cudaFree(d_cols_batch);
        cudaFree(d_lda_batch);
        cudaFree(d_ldb_batch);
        cudaFree(d_A_ptrs);
        cudaFree(d_B_ptrs);
        cudaFree(d_A);
        cudaFree(d_B);
        if(stop){
            break;
        }
        break;
    }
    #endif
    // cudaEventRecord(stopCode);
    // cudaEventSynchronize(stopCode);
    // float Code_time=0;
    // cudaEventElapsedTime(&Code_time, startCode, stopCode);
    // cudaEventDestroy(startCode);
    // cudaEventDestroy(stopCode);
    // printf("total time: %f\n", Code_time);
    // timer_arr[11] = Code_time;
    // printCountersInFile(timer_arr);
    // free(timer_arr);
}
