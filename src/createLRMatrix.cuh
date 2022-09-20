#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <cub/cub.cuh>
#include <assert.h>
#include <curand_kernel.h>

#include "magma_auxiliary.h"

#include "kblas.h"
#include "batch_rand.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// TODO: clean this file
uint64_t createLRMatrix(int n, int num_segments, int max_segment_size, int bucket_size, int dim, TLR_Matrix &matrix, H2Opus_Real* &d_denseMatrix, int* &d_values_in, int* &d_offsets_sort, H2Opus_Real* &d_dataset, float tolerance, int ARA_R, int max_rows, int max_cols, int max_rank){
    H2Opus_Real* d_input_matrix_segmented;
    gpuErrchk(cudaMalloc((void**) &d_input_matrix_segmented, max_segment_size*max_segment_size*num_segments*(uint64_t)sizeof(H2Opus_Real)));

    cudaMalloc((void**) &d_denseMatrix, num_segments*max_segment_size*num_segments*max_segment_size*sizeof(H2Opus_Real));

    int* d_scan_K_segmented;
    gpuErrchk(cudaMalloc((void**) &d_scan_K_segmented, (num_segments-1)*sizeof(int)));

    H2Opus_Real** d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    H2Opus_Real** d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    gpuErrchk(cudaMalloc((void**) &matrix.blockRanks, num_segments*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &matrix.diagonal, num_segments*max_segment_size*max_segment_size*sizeof(H2Opus_Real)));

    magma_init();

    int *d_rows_batch, *d_cols_batch, *d_ranks;
    int *d_ldm_batch, *d_lda_batch, *d_ldb_batch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real** d_M_ptrs, **d_A_ptrs, **d_B_ptrs;

    // TODO: fix memory allocation. Change num_segments to num_segments-1
    gpuErrchk(cudaMalloc((void**) &d_rows_batch, (num_segments-1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_cols_batch, (num_segments-1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ranks, (num_segments-1)*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldm_batch, (num_segments-1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_lda_batch, (num_segments-1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldb_batch, (num_segments-1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_A, (num_segments-1)*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_B, (num_segments-1)*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_M_ptrs, (num_segments-1)*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_A_ptrs, (num_segments-1)*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_B_ptrs, (num_segments-1)*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaPeekAtLastError());

    // TODO: parallelize
    int numThreadsPerBlock = 1024;
    int numBlocks = ((num_segments-1) + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays<<<numBlocks, numThreadsPerBlock>>>(num_segments-1, max_rows, max_cols, d_rows_batch, d_cols_batch, d_ldm_batch, d_lda_batch, d_ldb_batch);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kblasHandle_t kblas_handle;
    kblasRandState_t rand_state;
    kblasCreate(&kblas_handle);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kblasInitRandState(kblas_handle, &rand_state, 1<<15, 0);
    gpuErrchk(cudaPeekAtLastError());

    kblasEnableMagma(kblas_handle);
    kblas_gesvj_batch_wsquery<H2Opus_Real>(kblas_handle, max_rows, max_cols, num_segments-1);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle, bucket_size, num_segments-1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    kblasAllocateWorkspace(kblas_handle);
    cudaDeviceSynchronize();

    float ARATotalTime = 0;
    uint64_t k_sum = 0;

    dim3 m_numThreadsPerBlock(min(32, (int)max_segment_size), min(32, (int)max_segment_size));
    dim3 m_numBlocks(1, num_segments);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        // TODO: launch a 1D grid instead of a 2D grid
        #if 1
        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(n, num_segments, max_segment_size, dim, d_values_in, d_input_matrix_segmented, d_dataset, d_offsets_sort, segment, matrix.diagonal, d_denseMatrix, 1);
        #else
        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(n, num_segments, max_segment_size, dim, d_values_in, d_input_matrix_segmented, d_dataset, d_offsets_sort, segment, matrix.diagonal, d_denseMatrix, 0);
        #endif
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        
        generateArrayOfPointersT<H2Opus_Real>(d_input_matrix_segmented, d_M_ptrs, max_rows*max_cols, num_segments-1, 0);
        gpuErrchk(cudaPeekAtLastError());
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols, num_segments-1, 0);
        gpuErrchk(cudaPeekAtLastError());
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols, num_segments-1, 0);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        cudaDeviceSynchronize();

        cudaEvent_t startARA, stopARA;
        cudaEventCreate(&startARA);
        cudaEventCreate(&stopARA);
        cudaEventRecord(startARA);

        int kblas_ara_return = kblas_ara_batch(
            kblas_handle, d_rows_batch, d_cols_batch, d_M_ptrs, d_ldm_batch, 
            d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks + segment*(num_segments-1),
            tolerance, max_rows, max_cols, max_rank, 32, ARA_R, rand_state, 0, num_segments-1
        );
        assert(kblas_ara_return == 1);

        cudaEventRecord(stopARA);
        cudaEventSynchronize(stopARA);
        float ARA_time = 0;
        cudaEventElapsedTime(&ARA_time, startARA, stopARA);
        ARATotalTime += ARA_time;
        cudaEventDestroy(startARA);
        cudaEventDestroy(stopARA);
        cudaDeviceSynchronize();

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks + segment*(num_segments-1), d_scan_K_segmented, num_segments-1);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks + segment*(num_segments-1), d_scan_K_segmented, num_segments-1);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        printK<<<1, 1>>>(d_ranks + segment*(num_segments-1), num_segments - 1);

        uint64_t* totalMem = (uint64_t*)malloc(sizeof(uint64_t));
        uint64_t* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(uint64_t));
        getTotalMem<<<1, 1>>> (d_totalMem, d_ranks + segment*(num_segments-1), d_scan_K_segmented, num_segments-1);
        cudaDeviceSynchronize();
        cudaMemcpy(totalMem, d_totalMem, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_totalMem);

        gpuErrchk(cudaMalloc((void**) &d_U_tiled_temp[segment], max_segment_size*(*totalMem)*(uint64_t)sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_V_tiled_temp[segment], max_segment_size*(*totalMem)*(uint64_t)sizeof(H2Opus_Real)));

        // TODO: optimize thread allocation here
        int numThreadsPerBlock = max_segment_size;
        int numBlocks = num_segments-1;
        copyTiles<<<numBlocks, numThreadsPerBlock>>>(num_segments-1, max_segment_size, d_ranks + segment*(num_segments-1), d_scan_K_segmented, d_U_tiled_temp[segment], d_A, d_V_tiled_temp[segment], d_B);
        cudaDeviceSynchronize();

        k_sum += (*totalMem);
        free(totalMem);
    }

    cudaFree(d_input_matrix_segmented);
    cudaFree(d_scan_K_segmented);
    cudaFree(d_values_in);
    cudaFree(d_offsets_sort);
    cudaFree(d_dataset);

    cudaFree(d_rows_batch);
    cudaFree(d_cols_batch);
    cudaFree(d_ldm_batch);
    cudaFree(d_lda_batch);
    cudaFree(d_ldb_batch);
    cudaFree(d_M_ptrs);
    cudaFree(d_A_ptrs);
    cudaFree(d_B_ptrs);
    cudaFree(d_A);
    cudaFree(d_B);

    gpuErrchk(cudaMalloc((void**) &matrix.U, k_sum*max_segment_size*(uint64_t)sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix.V, k_sum*max_segment_size*(uint64_t)sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix.blockOffsets, num_segments*num_segments*(uint64_t)sizeof(int)));

    numThreadsPerBlock = 1024;
    numBlocks = ((num_segments-1)*num_segments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    copyRanks<<<numBlocks, numThreadsPerBlock>>>(num_segments, max_segment_size, d_ranks, matrix.blockRanks);
    cudaDeviceSynchronize();
    cudaFree(d_ranks);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);
    gpuErrchk(cudaPeekAtLastError());

    // printK<<<1, 1>>>(matrix.blockRanks, num_segments*num_segments);
    // printK<<<1, 1>>>(matrix.blockOffsets, num_segments*num_segments);

    int* h_scan_K = (int*)malloc(num_segments*num_segments*sizeof(int));
    gpuErrchk(cudaMemcpy(h_scan_K, matrix.blockOffsets, num_segments*num_segments*(uint64_t)sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

    for(unsigned int segment = 0; segment < num_segments-1; ++segment){
        gpuErrchk(cudaMemcpy(&matrix.U[h_scan_K[num_segments*segment]*max_segment_size], d_U_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*max_segment_size*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(&matrix.V[h_scan_K[num_segments*segment]*max_segment_size], d_V_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*max_segment_size*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    }
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(&matrix.U[h_scan_K[num_segments*(num_segments-1)]*max_segment_size], d_U_tiled_temp[num_segments-1], (k_sum - h_scan_K[num_segments*(num_segments-1)])*max_segment_size*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(&matrix.V[h_scan_K[num_segments*(num_segments-1)]*max_segment_size], d_V_tiled_temp[num_segments-1], (k_sum - h_scan_K[num_segments*(num_segments-1)])*max_segment_size*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    free(h_scan_K);
    gpuErrchk(cudaPeekAtLastError());

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        cudaFree(d_U_tiled_temp[segment]);
        cudaFree(d_V_tiled_temp[segment]);
    }
    free(d_U_tiled_temp);
    free(d_V_tiled_temp);
    gpuErrchk(cudaPeekAtLastError());

    return k_sum;
}