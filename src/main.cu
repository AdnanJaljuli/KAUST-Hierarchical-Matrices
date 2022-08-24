#include "tlr_example.h"
#include "TLR_Matrix.cuh"
#include "helperFunctions.h"
#include "helperKernels.cuh"
#include "config.h"
#include "kdtreeConstruction.cuh"
#include "magma_auxiliary.h"

#include "kblas.h"
#include "batch_rand.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <typeinfo>
#include <algorithm>
#include <string.h>
#include <stdio.h>

#define EXPAND_MATRIX 0
#define BLOCK_SIZE 32
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

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

    float* timer_arr = (float*)malloc(numTimers*sizeof(float));
    timer_arr[0] = (float)config.n;
    timer_arr[1] = (float)config.bucket_size;
    timer_arr[2] = (float)config.dim;
    timer_arr[3] = (float)config.tol;

    cudaError_t cudaErr;
    H2Opus_Real *d_dataset;
    gpuErrchk(cudaMalloc((void**) &d_dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real)));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (config.n+numThreadsPerBlock-1)/numThreadsPerBlock;
    generateDataset<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, d_dataset);
    cudaDeviceSynchronize();

    uint64_t num_segments = 1;
    int max_num_segments;
    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        max_num_segments = 1<<(getMaxSegmentSize(config.n, config.bucket_size).second);
    } else {
        max_num_segments = (config.n+config.bucket_size-1)/config.bucket_size;
    }

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

    uint64_t maxSegmentSize;
    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        maxSegmentSize = getMaxSegmentSize(config.n, config.bucket_size).first;
    } else {
        maxSegmentSize = config.bucket_size;
    }
    printf("max segment size: %lu\n", maxSegmentSize);

    H2Opus_Real* d_input_matrix_segmented;

    printf("mem allocated to input matrix: %lu\n", maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    gpuErrchk(cudaMalloc((void**) &d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real)));

    int* d_scan_K_segmented;
    gpuErrchk(cudaMalloc((void**) &d_scan_K_segmented, (num_segments-1)*sizeof(int)));

    H2Opus_Real** d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    H2Opus_Real** d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    TLR_Matrix matrix;
    gpuErrchk(cudaMalloc((void**) &matrix.blockRanks, num_segments*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real)));

    magma_init();

    const int ARA_R = 10;
    const int max_rows = maxSegmentSize;
    const int max_cols = maxSegmentSize;
    const int max_rank = max_cols;

    int *d_rows_batch, *d_cols_batch, *d_ranks;
    int *d_ldm_batch, *d_lda_batch, *d_ldb_batch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real** d_M_ptrs, **d_A_ptrs, **d_B_ptrs;

    // TODO: fix memory allocation. Change num_segments to num_segments-1
    gpuErrchk(cudaMalloc((void**) &d_rows_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_cols_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ranks, (num_segments-1)*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldm_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_lda_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldb_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_A, num_segments*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_B, num_segments*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_M_ptrs, num_segments*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_A_ptrs, num_segments*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_B_ptrs, num_segments*sizeof(H2Opus_Real*)));

    numThreadsPerBlock = 1024;
    numBlocks = ((num_segments-1) + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays<<<1, 1>>>(num_segments-1, max_rows, max_cols, d_rows_batch, d_cols_batch, d_ldm_batch, d_lda_batch, d_ldb_batch);
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
    kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle, config.bucket_size, num_segments-1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    kblasAllocateWorkspace(kblas_handle);
    cudaDeviceSynchronize();
    
    float ARATotalTime = 0;
    int k_sum = 0;

    #if EXPAND_MATRIX
    H2Opus_Real* d_error;
    H2Opus_Real* error = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));

    H2Opus_Real* d_tmp;
    H2Opus_Real* tmp = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));

    *error = 0;
    *tmp = 0;
    cudaMemcpy(d_error, error, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp, tmp, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

    H2Opus_Real* d_expMatrix;
    gpuErrchk(cudaMalloc((void**) &d_expMatrix, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real)));
    #endif

    cudaEvent_t startGenerateInputMatrix, stopGenerateInputMatrix;
    cudaEventCreate(&startGenerateInputMatrix);
    cudaEventCreate(&stopGenerateInputMatrix);
    cudaEventRecord(startGenerateInputMatrix);

    dim3 m_numThreadsPerBlock(min(32, (int)maxSegmentSize), min(32, (int)maxSegmentSize));
    dim3 m_numBlocks(1, num_segments);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(config.n, num_segments, maxSegmentSize, config.dim, d_values_in, d_input_matrix_segmented, d_dataset, d_offsets_sort, segment, matrix.diagonal);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        #if 0
        H2Opus_Real* input_matrix_segmented = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*(num_segments-1)*(uint64_t)sizeof(H2Opus_Real));
        cudaMemcpy(input_matrix_segmented, d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*(num_segments-1)*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        char filename[100] = "results/inputmatrix.txt";
        FILE *output_file = fopen(filename, "a");
        for(unsigned int i=0; i<num_segments-1; ++i){
            for(unsigned int j=0; j<maxSegmentSize; ++j){
                for(unsigned int k=0; k<maxSegmentSize; ++k){
                    // fprintf(output_file,"%lf ", input_matrix_segmented[i*maxSegmentSize*maxSegmentSize + k*maxSegmentSize + j]);
                    printf("%lf ", input_matrix_segmented[i*maxSegmentSize*maxSegmentSize + k*maxSegmentSize + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
        fclose(output_file);
        free(input_matrix_segmented);
        #endif

        int* totalMem = (int*)malloc(sizeof(int));
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
            config.tol, max_rows, max_cols, max_rank, 32, ARA_R, rand_state, 0, num_segments-1
        );
        printf("kblas_ara_return: %d\n", kblas_ara_return);

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

        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, d_ranks + segment*(num_segments-1), d_scan_K_segmented, num_segments-1);
        cudaDeviceSynchronize();
        cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_totalMem);

        #if 1
        printARAOutput<<<1, 1>>>(d_A, d_B, d_ranks + segment*(num_segments-1), num_segments-1, max_rows, max_rank);
        #endif
        gpuErrchk(cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));

        numThreadsPerBlock = maxSegmentSize;
        numBlocks = num_segments-1;
        copyTiles<<<numBlocks, numThreadsPerBlock>>>(num_segments-1, maxSegmentSize, d_ranks + segment*(num_segments-1), d_scan_K_segmented, d_U_tiled_temp[segment], d_A, d_V_tiled_temp[segment], d_B);
        cudaDeviceSynchronize();

        #if EXPAND_MATRIX
        expandMatrix<<<m_numBlocks, m_numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_ranks + segment*num_segments, d_scan_K_segmented, d_U_tiled_temp[segment], d_V_tiled_temp[segment], d_expMatrix);
        cudaDeviceSynchronize();

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments*maxSegmentSize*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
        calcError<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix_segmented, d_error, d_tmp);
        cudaDeviceSynchronize();
        #endif

        k_sum += (*totalMem);
        free(totalMem);
    }
    timer_arr[5] = k_sum;
    cudaDeviceSynchronize();
    cudaEventRecord(stopGenerateInputMatrix);
    cudaEventSynchronize(stopGenerateInputMatrix);
    cudaEventElapsedTime(&timer_arr[6], startGenerateInputMatrix, stopGenerateInputMatrix);
    cudaEventDestroy(startGenerateInputMatrix);
    cudaEventDestroy(stopGenerateInputMatrix);

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

    #if EXPAND_MATRIX
    cudaMemcpy(error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaFree(d_error);
    cudaFree(d_tmp);
    printf("error: %lf\n", sqrt(*error)/sqrt(*tmp));
    timer_arr[12] = sqrt(*error)/sqrt(*tmp);
    free(tmp);
    free(error);
    cudaFree(d_expMatrix);
    #endif

    printf("k sum: %d\n", k_sum);
    gpuErrchk(cudaMalloc((void**) &matrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix.blockOffsets, num_segments*num_segments*sizeof(int)));

    numThreadsPerBlock = 1024;
    numBlocks = ((num_segments-1)*num_segments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    copyRanks<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, d_ranks, matrix.blockRanks);
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

    int* h_scan_K = (int*)malloc(num_segments*num_segments*sizeof(int));
    gpuErrchk(cudaMemcpy(h_scan_K, matrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

    for(unsigned int segment = 0; segment < num_segments-1; ++segment){
        gpuErrchk(cudaMemcpy(&matrix.U[h_scan_K[num_segments*segment]*maxSegmentSize], d_U_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(&matrix.V[h_scan_K[num_segments*segment]*maxSegmentSize], d_V_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    }
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(&matrix.U[h_scan_K[num_segments*(num_segments-1)]*maxSegmentSize], d_U_tiled_temp[num_segments-1], (k_sum - h_scan_K[num_segments*(num_segments-1)])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(&matrix.V[h_scan_K[num_segments*(num_segments-1)]*maxSegmentSize], d_V_tiled_temp[num_segments-1], (k_sum - h_scan_K[num_segments*(num_segments-1)])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    free(h_scan_K);
    gpuErrchk(cudaPeekAtLastError());

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        cudaFree(d_U_tiled_temp[segment]);
        cudaFree(d_V_tiled_temp[segment]);
    }
    free(d_U_tiled_temp);
    free(d_V_tiled_temp);
    gpuErrchk(cudaPeekAtLastError());

    TLR_Matrix mortonMatrix;
    cudaMalloc((void**) &mortonMatrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.blockOffsets, num_segments*num_segments*sizeof(int));
    cudaMalloc((void**) &mortonMatrix.blockRanks, num_segments*num_segments*sizeof(int));
    cudaMalloc((void**) &mortonMatrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    ColumnMajorToMorton(num_segments, maxSegmentSize, k_sum, matrix, mortonMatrix);
    gpuErrchk(cudaPeekAtLastError());
    printf("k_sum %d\n", k_sum);


    const int num_ops = (num_segments/2)*(num_segments/2) - (num_segments/2);

    int* d_ranks_output;
    gpuErrchk(cudaMalloc((void**) &d_ranks_output, num_ops*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_rows_batch, num_ops*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_cols_batch, num_ops*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_lda_batch, num_ops*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldb_batch, num_ops*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_A_ptrs, num_ops*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_B_ptrs, num_ops*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_A, num_ops*2*max_rows*2*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_B, num_ops*2*max_rows*2*max_rank*sizeof(H2Opus_Real)));

    gpuErrchk(cudaPeekAtLastError());
    fillARAArrays_mod<<<1, 1>>>(num_ops, max_rows, max_cols, d_rows_batch, d_cols_batch, d_lda_batch, d_ldb_batch);
    gpuErrchk(cudaPeekAtLastError());
    printf("fillARAArrays_mod\n");
    generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols*4, num_ops, 0);
    printf("first\n");
    generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols*4, num_ops, 0);
    printf("second\n");
    gpuErrchk(cudaPeekAtLastError());

    kblasHandle_t kblas_handle_2;
    kblasRandState_t rand_state_2;
    kblasCreate(&kblas_handle_2);
    cudaDeviceSynchronize();
    printf("kblas handle 2\n");
    gpuErrchk(cudaPeekAtLastError());

    int kblasrandstate_value = kblasInitRandState(kblas_handle_2, &rand_state_2, 1<<15, 0);
    printf("kblasinitrandstate: %d\n", kblasrandstate_value);
    gpuErrchk(cudaPeekAtLastError());

    kblasEnableMagma(kblas_handle_2);
    kblas_gesvj_batch_wsquery<H2Opus_Real>(kblas_handle_2, 2*max_rows, 2*max_cols, num_ops);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle_2, config.bucket_size, num_ops);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    kblasAllocateWorkspace(kblas_handle_2);
    cudaDeviceSynchronize();

    int* d_activeArrays;
    int* d_tmpArray;
    int* d_ranks_1;
    gpuErrchk(cudaMalloc((void**) &d_ranks_1, num_ops*4*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_activeArrays, (num_segments*num_segments)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_tmpArray, (num_segments*num_segments)*sizeof(int)));
    fillActiveArrays<<<1, 1>>>(num_segments, d_activeArrays, d_tmpArray, d_ranks_1, mortonMatrix.blockRanks);
    cudaFree(d_tmpArray);

    printf("num ops: %d\n", num_ops);
    gpuErrchk(cudaPeekAtLastError());

    int kblas_mod_ans = kblas_ara_batch_mod(kblas_handle_2, d_rows_batch, d_cols_batch, mortonMatrix.U, mortonMatrix.V, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, d_activeArrays,
        d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks_output,
        config.tol, 2*max_rows, 2*max_cols, 2*max_rank, 32, ARA_R, rand_state_2, 0, ((num_segments/2)*(num_segments/2) - (num_segments/2))
    );
    printf("kblas mod ans : %d\n", kblas_mod_ans);
    
    printK<<<1, 1>>>(d_ranks_output, num_ops);
    cudaDeviceSynchronize();
    printK<<<1, 1>>>(mortonMatrix.blockRanks, num_segments*num_segments);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    #if 0
    int* d_bit_vector;
    int* d_bit_vector_scan;
    gpuErrchk(cudaMalloc((void**) &d_bit_vector, num_ops*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_bit_vector_scan, num_ops*sizeof(int)));

    numThreadsPerBlock = 1024;
    numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_ops, 32, d_ranks_output, d_ranks_1, d_bit_vector);
    cudaDeviceSynchronize();

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bit_vector, d_bit_vector_scan, num_ops);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bit_vector, d_bit_vector_scan, num_ops);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    int* d_new_ranks;
    int* d_new_active_tiles;
    cudaMalloc((void**) &d_new_ranks, num_ops*sizeof(int));
    cudaMalloc((void**) &d_new_active_tiles, num_ops*sizeof(int));

    numThreadsPerBlock = 1024;
    numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillNewLevel<<<numBlocks, numThreadsPerBlock>>>(num_ops, d_bit_vector, d_bit_vector_scan, d_ranks_output, d_new_ranks, d_new_active_tiles);
    #endif 

    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float Code_time=0;
    cudaEventElapsedTime(&Code_time, startCode, stopCode);
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    printf("total time: %f\n", Code_time);
    timer_arr[11] = Code_time;
    printCountersInFile(timer_arr);
    free(timer_arr);
}