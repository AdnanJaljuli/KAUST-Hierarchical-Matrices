#include "tlr_example.h"
#include "TLR_Matrix.cuh"
#include "helperFunctions.h"
#include "helperKernels.cuh"
#include "config.h"
#include "kdtreeConstruction.cuh"

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
#include "cublas_v2.h"

#include "kblas.h"
#include "batch_rand.h"
#include "batch_pstrf.h"
#include "batch_block_copy.h"
#include "batch_ara.h"
#include "magma_auxiliary.h"

#define EXPAND_MATRIX 1
#define DENSE_CALC 1
#define BLOCK_SIZE 32
#define PRINT_OUTPUT 0
#define KBLAS_ARA 1
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

// TODO: make sure that non powers of two work
// TODO: create a struct for the tiled matrix that has u_tiled, v_tiled, k, k_scan
// TODO: generate pointcloud and copy values of the pointcloud to ptr on GPU
// TODO: fix makefile so main.cu depends on helperKerlens.cuh
// TODO: make sure that everything that is malloced is freed
// TODO: move generate input matrix to its own file
// TODO: move the expand matrix function to outside the for loop

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
    gpuErrchk(cudaMalloc((void**) &d_scan_K_segmented, num_segments*sizeof(int)));

    H2Opus_Real** d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    H2Opus_Real** d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    TLR_Matrix matrix;
    gpuErrchk(cudaMalloc((void**) &matrix.blockRanks, num_segments*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real)));

    printf("ARA begins\n");
    magma_init();

    const int ARA_R = 10;
    const int max_rows = maxSegmentSize;
    const int max_cols = maxSegmentSize;
    const int max_rank = max_cols;

    int *d_rows_batch, *d_cols_batch, *d_ranks;
    int *d_ldm_batch, *d_lda_batch, *d_ldb_batch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real** d_M_ptrs, **d_A_ptrs, **d_B_ptrs;

    gpuErrchk(cudaMalloc((void**) &d_rows_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_cols_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ranks, num_segments*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldm_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_lda_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldb_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_A, num_segments*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_B, num_segments*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_M_ptrs, num_segments*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_A_ptrs, num_segments*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_B_ptrs, num_segments*sizeof(H2Opus_Real*)));

    numThreadsPerBlock = 1024;
    numBlocks = (num_segments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays<<<1, 1>>>(num_segments, max_rows, max_cols, d_rows_batch, d_cols_batch, d_ldm_batch, d_lda_batch, d_ldb_batch);
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
    kblas_gesvj_batch_wsquery<H2Opus_Real>(kblas_handle, max_rows, max_cols, num_segments);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle, config.bucket_size, num_segments);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    kblasAllocateWorkspace(kblas_handle);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    
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

        #if 1
        H2Opus_Real* input_matrix_segmented = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
        cudaMemcpy(input_matrix_segmented, d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        char filename[100] = "results/inputmatrix.txt";
        FILE *output_file = fopen(filename, "a");
        for(unsigned int i=0; i<num_segments; ++i){
            for(unsigned int j=0; j<maxSegmentSize; ++j){
                for(unsigned int k=0; k<maxSegmentSize; ++k){
                    fprintf(output_file,"%lf ", input_matrix_segmented[i*maxSegmentSize*maxSegmentSize + k*maxSegmentSize + j]);
                }
                fprintf(output_file, "\n");
            }
            fprintf(output_file, "\n");
        }
        fprintf(output_file, "\n");
        fclose(output_file);
        free(input_matrix_segmented);
        #endif

        int* totalMem = (int*)malloc(sizeof(int));
        generateArrayOfPointersT<H2Opus_Real>(d_input_matrix_segmented, d_M_ptrs, max_rows*max_cols, num_segments, 0);
        gpuErrchk(cudaPeekAtLastError());
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols, num_segments, 0);
        gpuErrchk(cudaPeekAtLastError());
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols, num_segments, 0);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        cudaDeviceSynchronize();

        cudaEvent_t startARA, stopARA;
        cudaEventCreate(&startARA);
        cudaEventCreate(&stopARA);
        cudaEventRecord(startARA);
        kblas_ara_batch(
                            kblas_handle, d_rows_batch, d_cols_batch, d_M_ptrs, d_ldm_batch, 
                            d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks + segment*num_segments, 
                            config.tol, max_rows, max_cols, max_rank, 32, ARA_R, rand_state, 0, num_segments
                        );
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
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, d_ranks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_totalMem);

        #if 1
        printARAOutput<<<1, 1>>>(d_A, d_B, d_ranks + segment*num_segments, num_segments, max_rows, max_rank);
        #endif
        gpuErrchk(cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));

        numThreadsPerBlock = maxSegmentSize;
        numBlocks = num_segments;
        copyTiles<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, d_ranks + segment*num_segments, d_scan_K_segmented, d_U_tiled_temp[segment], d_A, d_V_tiled_temp[segment], d_B);
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
    // printf("total mem %d\n", k_sum);
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

    #if KBLAS_ARA
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
    #endif

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

    gpuErrchk(cudaMemcpy(matrix.blockRanks, d_ranks, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToDevice));
    cudaFree(d_ranks);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    int* h_scan_K = (int*)malloc(num_segments*num_segments*sizeof(int));
    gpuErrchk(cudaMemcpy(h_scan_K, matrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToHost));

    for(unsigned int segment = 0; segment < num_segments-1; ++segment){
        gpuErrchk(cudaMemcpy(&matrix.U[h_scan_K[num_segments*segment]*maxSegmentSize], d_U_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(&matrix.V[h_scan_K[num_segments*segment]*maxSegmentSize], d_V_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    }
    gpuErrchk(cudaMemcpy(&matrix.U[h_scan_K[num_segments*(num_segments-1)]*maxSegmentSize], d_U_tiled_temp[num_segments-1], (k_sum - h_scan_K[num_segments*(num_segments-1)])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(&matrix.V[h_scan_K[num_segments*(num_segments-1)]*maxSegmentSize], d_V_tiled_temp[num_segments-1], (k_sum - h_scan_K[num_segments*(num_segments-1)])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    free(h_scan_K);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        cudaFree(d_U_tiled_temp[segment]);
        cudaFree(d_V_tiled_temp[segment]);
    }
    free(d_U_tiled_temp);
    free(d_V_tiled_temp);

    H2Opus_Real* d_buffer_vector;
    H2Opus_Real* d_input_vector;
    H2Opus_Real* d_output_vector;
    H2Opus_Real* d_output_vector_org;
    gpuErrchk(cudaMalloc((void**) &d_buffer_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_input_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_output_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_output_vector_org, maxSegmentSize*num_segments*sizeof(H2Opus_Real)));

    numThreadsPerBlock = 1024;
    numBlocks = (num_segments*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
    fillVector<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_input_vector, d_output_vector, d_output_vector_org);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaEvent_t startGEMV, stopGEMV;
    cudaEventCreate(&startGEMV);
    cudaEventCreate(&stopGEMV);
    cudaEventRecord(startGEMV);
    numThreadsPerBlock = maxSegmentSize;
    numBlocks = num_segments;
    GEMV<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, matrix.blockRanks, matrix.blockOffsets, matrix.U, matrix.V, matrix.diagonal, d_input_vector, d_output_vector, d_buffer_vector);
    cudaEventRecord(stopGEMV);
    cudaEventSynchronize(stopGEMV);
    gpuErrchk(cudaPeekAtLastError());
    cudaEventElapsedTime(&timer_arr[7], startGEMV, stopGEMV);
    cudaEventDestroy(startGEMV);
    cudaEventDestroy(stopGEMV);

    cudaFree(d_buffer_vector);
    cudaFree(d_input_vector);
    cudaFree(d_output_vector);
    cudaFree(d_output_vector_org);

    #if DENSE_CALC
    H2Opus_Real* d_denseMatrix;
    H2Opus_Real* d_vector;
    gpuErrchk(cudaMalloc((void**) &d_denseMatrix, config.n*config.n*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_vector, config.n*sizeof(H2Opus_Real)));

    numThreadsPerBlock = 1024;
    numBlocks = (config.n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillDenseMatrix<<<numBlocks, numThreadsPerBlock>>>(config.n, d_denseMatrix);
    filltmpVector<<<numBlocks, numThreadsPerBlock>>>(config.n, d_vector);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    H2Opus_Real alfa=1, beta=0;
    
    cudaEvent_t startDenseGEMV, stopDenseGEMV;
    cudaEventCreate(&startDenseGEMV);
    cudaEventCreate(&stopDenseGEMV);
    cudaEventRecord(startDenseGEMV);
    cublasDgemv(handle, CUBLAS_OP_T,
                           config.n, config.n,
                           &alfa,
                           d_denseMatrix, config.n,
                           d_vector, 1,
                           &beta,
                           d_vector, 1);
    cudaDeviceSynchronize();
    cudaEventRecord(stopDenseGEMV);
    cudaEventSynchronize(stopDenseGEMV);
    cudaEventElapsedTime(&timer_arr[8], startDenseGEMV, stopDenseGEMV);
    cudaEventDestroy(startDenseGEMV);
    cudaEventDestroy(stopDenseGEMV);

    cublasDestroy(handle);
    cudaFree(d_denseMatrix);
    cudaFree(d_vector);
    #endif

    #if 1
    TLR_Matrix matrix2;
    gpuErrchk(cudaMalloc((void**) &matrix2.blockRanks, num_segments*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &matrix2.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix2.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix2.diagonal, maxSegmentSize*maxSegmentSize*num_segments*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix2.blockOffsets, num_segments*num_segments*sizeof(int)));

    cudaMemcpy(matrix2.blockRanks, matrix.blockRanks, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.blockOffsets, matrix.blockOffsets, num_segments*num_segments*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.U, matrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.V, matrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.diagonal, matrix.diagonal, maxSegmentSize*maxSegmentSize*num_segments*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

    int* d_scan_K_output;

    gpuErrchk(cudaMalloc((void**) &d_scan_K_output, num_segments*num_segments*sizeof(unsigned int)));

    H2Opus_Real* d_gemm_matrix_segmented;
    gpuErrchk(cudaMalloc((void**) &d_gemm_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real)));

    d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    gpuErrchk(cudaMalloc((void**) &d_scan_K_segmented, num_segments*sizeof(int)));
    k_sum = 0;
    ARATotalTime = 0;

    gpuErrchk(cudaMalloc((void**) &d_rows_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_cols_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ranks, num_segments*num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldm_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_lda_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ldb_batch, num_segments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_A, num_segments*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_B, num_segments*max_rows*max_rank*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_M_ptrs, num_segments*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_A_ptrs, num_segments*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_B_ptrs, num_segments*sizeof(H2Opus_Real*)));

    numThreadsPerBlock = maxSegmentSize;
    numBlocks = num_segments;
    fillARAArrays<<<1, 1>>>(num_segments, max_rows, max_cols, d_rows_batch, d_cols_batch, d_ldm_batch, d_lda_batch, d_ldb_batch);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaEvent_t startGEMM, stopGEMM;
    cudaEventCreate(&startGEMM);
    cudaEventCreate(&stopGEMM);
    cudaEventRecord(startGEMM);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        GEMM<<<m_numBlocks, m_numThreadsPerBlock, 2*config.bucket_size*config.bucket_size*sizeof(H2Opus_Real)>>>(num_segments, maxSegmentSize, matrix.U, matrix.V, matrix.diagonal, matrix.blockRanks, matrix.blockOffsets, matrix2.U, matrix2.V, matrix2.diagonal, matrix2.blockRanks, matrix2.blockOffsets, d_gemm_matrix_segmented, segment, config.bucket_size);
        cudaDeviceSynchronize();

        int* totalMem = (int*)malloc(sizeof(int));
        generateArrayOfPointersT<H2Opus_Real>(d_gemm_matrix_segmented, d_M_ptrs, max_rows*max_cols, num_segments, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols, num_segments, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols, num_segments, 0);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        cudaDeviceSynchronize();

        cudaEvent_t startARA, stopARA;
        cudaEventCreate(&startARA);
        cudaEventCreate(&stopARA);
        cudaEventRecord(startARA);
        kblas_ara_batch(
                            kblas_handle, d_rows_batch, d_cols_batch, d_M_ptrs, d_ldm_batch, 
                            d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks + segment*num_segments, 
                            config.tol, max_rows, max_cols, max_rank, 32, ARA_R, rand_state, 0, num_segments
                        );
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
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, d_ranks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        gpuErrchk(cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_totalMem);

        #if 1
        printARAOutput<<<1, 1>>>(d_A, d_B, d_ranks + segment*num_segments, num_segments, max_rows, max_rank);
        #endif

        gpuErrchk(cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));

        // copy d_A and d_B to d_U_tiled_segmented
        numThreadsPerBlock = maxSegmentSize;
        numBlocks = num_segments;
        copyTiles<<<numBlocks, numThreadsPerBlock>>>(num_segments, maxSegmentSize, d_ranks+ segment*num_segments, d_scan_K_segmented, d_U_tiled_temp[segment], d_A, d_V_tiled_temp[segment], d_B);
        cudaDeviceSynchronize();

        k_sum += (*totalMem);
        free(totalMem);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stopGEMM);
    cudaEventSynchronize(stopGEMM);
    float GEMM_time = 0;
    cudaEventElapsedTime(&GEMM_time, startGEMM, stopGEMM);
    cudaEventDestroy(startGEMM);
    cudaEventDestroy(stopGEMM);
    printf("GEMM time: %f\n", GEMM_time);
    timer_arr[9] = GEMM_time;
    magma_finalize();
    // TODO: copy from double pointer array to a single pointer 

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        cudaFree(d_U_tiled_temp[segment]);
        cudaFree(d_V_tiled_temp[segment]);
    }
    free(d_U_tiled_temp);
    free(d_V_tiled_temp);

    #if DENSE_CALC
    H2Opus_Real* d_denseMatrix1;
    cudaErr = cudaMalloc((void**) &d_denseMatrix1, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error dense matrix 1: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_denseMatrix2;
    cudaErr = cudaMalloc((void**) &d_denseMatrix2, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error dense matrix 2: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_denseMatrix3;
    cudaErr = cudaMalloc((void**) &d_denseMatrix3, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error dense matrix3: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024;
    numBlocks = (config.n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillDenseMatrix<<<numBlocks, numThreadsPerBlock>>>(config.n, d_denseMatrix1);
    fillDenseMatrix<<<numBlocks, numThreadsPerBlock>>>(config.n, d_denseMatrix2);
    cudaDeviceSynchronize();
    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cudaEvent_t startDenseGEMM, stopDenseGEMM;
    cudaEventCreate(&startDenseGEMM);
    cudaEventCreate(&stopDenseGEMM);
    cudaEventRecord(startDenseGEMM);
    cudaDeviceSynchronize();
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                           config.n, config.n, config.n,
                           &alfa,
                           d_denseMatrix1, config.n,
                           d_denseMatrix2, config.n,
                           &beta,
                           d_denseMatrix3, config.n);
    cudaDeviceSynchronize();
    cudaEventRecord(stopDenseGEMM);
    cudaEventSynchronize(stopDenseGEMM);
    float DenseGEMM_time = 0;
    cudaEventElapsedTime(&DenseGEMM_time, startDenseGEMM, stopDenseGEMM);
    printf("DenseGEMM time: %f\n", DenseGEMM_time);
    timer_arr[10] = DenseGEMM_time;
    cudaEventDestroy(startDenseGEMM);
    cudaEventDestroy(stopDenseGEMM);

    cublasDestroy(handle);
    cudaFree(d_denseMatrix1);
    cudaFree(d_denseMatrix2);
    cudaFree(d_denseMatrix3);
    #endif

    cudaFree(d_scan_K_segmented);
    cudaFree(d_gemm_matrix_segmented);
    
    cudaFreeMatrix(matrix);
    cudaFreeMatrix(matrix2);

    #endif
    cudaDeviceSynchronize();
    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float code_time = 0;
    float Code_time=0;
    cudaEventElapsedTime(&Code_time, startCode, stopCode);
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    printf("total time: %f\n", Code_time);
    timer_arr[11] = Code_time;
    printCountersInFile(timer_arr);
    free(timer_arr);
}