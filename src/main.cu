#include "tlr_example.h"
#include "TLR_Matrix.cuh"
#include "helperFunctions.h"
#include "helperKernels.cuh"
#include "SVD.cuh"
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

#define EXPAND_MATRIX 0
#define BLOCK_SIZE 32
#define PRINT_OUTPUT 0
#define KBLAS_ARA 0
using namespace std;

// TODO: make sure that non powers of two work
// TODO: create a struct for the tiled matrix that has u_tiled, v_tiled, k, k_scan
// TODO: generate pointcloud and copy values of the pointcloud to ptr on GPU
// TODO: fix makefile so main.cu depends on helperKerlens.cuh
// TODO: make sure that everything that is malloced is freed
// TODO: move kdtree contruction to its own file

int main(int argc, char *argv[]){
    cudaEvent_t startCode, stopCode;
    cudaEventCreate(&startCode);
    cudaEventCreate(&stopCode);
    cudaEventRecord(startCode);

    Config config = parseArgs(argc, argv);
    printf("n: %d\n", config.n);
    printf("bucket size: %d\n", config.bucket_size);
    printf("epsilon: %f\n", config.epsilon);
    printf("dim: %d\n", config.dim);

    float* timer_arr = (float*)malloc(numTimers*sizeof(float));
    timer_arr[0] = (float)config.n;
    timer_arr[1] = (float)config.bucket_size;
    timer_arr[2] = (float)config.dim;
    timer_arr[3] = (float)config.epsilon;

    // // _________________replace with generating the dataset on the GPU__________________________________
    // // Create point cloud
    // PointCloud<H2Opus_Real> pt_cloud(config.dim, (size_t)config.n);
    // generateGrid<H2Opus_Real>(pt_cloud, config.n);

    // H2Opus_Real *dataset;
    // dataset = (H2Opus_Real*)malloc(config.n*config.dim*(uint64_t)sizeof(H2Opus_Real));
    // assert(dataset != NULL);

    // // TODO: move this to a kernel
    // for (unsigned long long i = 0; i < config.dim; ++i){
    //     for(unsigned long long j = 0; j < config.n; ++j){
    //         dataset[i*config.n+j] = pt_cloud.getDataPoint((size_t)j, (int)i);
    //     }
    // }

    // H2Opus_Real *d_dataset;
    // cudaError_t cudaErr = cudaMalloc((void**) &d_dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real));
    // if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    // cudaErr = cudaMemcpy(d_dataset, dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    // if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    // free(dataset);
    // // ___________________________________________________________________________________________________
    H2Opus_Real *d_dataset;
    cudaError_t cudaErr = cudaMalloc((void**) &d_dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (config.n+numThreadsPerBlock-1)/numThreadsPerBlock;
    generateDataset<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, d_dataset);
    cudaDeviceSynchronize();

    uint64_t num_segments = 1;

    // _________________________________________________________________________________________________________________________
    int max_num_segments;
    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        max_num_segments = 1<<(getMaxSegmentSize(config.n, config.bucket_size).second);
    } else {
        max_num_segments = (config.n+config.bucket_size-1)/config.bucket_size;
    }
    printf("max num segments: %d\n", max_num_segments);
    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *d_offsets_sort;      // e.g., [-, -, -, -, -, -, -]
    cudaErr = cudaMalloc((void**) &d_offsets_sort, (max_num_segments + 1)*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_values_in, config.n*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    cudaEvent_t startKDtree, stopKDtree;
    cudaEventCreate(&startKDtree);
    cudaEventCreate(&stopKDtree);
    cudaEventRecord(startKDtree);
    createKDTree(config.n, config.dim, config.bucket_size, num_segments, config.div_method, d_values_in, d_offsets_sort, d_dataset, max_num_segments);
    cudaEventRecord(stopKDtree);
    cudaEventSynchronize(stopKDtree);
    float KDtree_time = 0;
    cudaEventElapsedTime(&KDtree_time, startKDtree, stopKDtree);
    timer_arr[4] = KDtree_time;
    cudaEventDestroy(startKDtree);
    cudaEventDestroy(stopKDtree);
    cudaDeviceSynchronize();
    printf("num segments: %d\n", num_segments);
    // _________________________________________________________________________________________________________________________

    uint64_t maxSegmentSize;
    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        maxSegmentSize = getMaxSegmentSize(config.n, config.bucket_size).first;
    } else {
        maxSegmentSize = config.bucket_size;
    }
    printf("max segment size: %lu\n", maxSegmentSize);

    H2Opus_Real* d_input_matrix_segmented;
    H2Opus_Real* input_matrix_segmented = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    printf("mem allocated to input matrix: %lu\n", maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    cudaErr = cudaMalloc((void**) &d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    H2Opus_Real *h_S = (H2Opus_Real *)malloc(maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    H2Opus_Real *h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    H2Opus_Real *h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));

    H2Opus_Real* d_S;
    H2Opus_Real* d_U;
    H2Opus_Real* d_V;
    cudaErr = cudaMalloc((void**) &d_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_U, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_V, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    int* d_scan_K_segmented;
    cudaErr = cudaMalloc((void**) &d_scan_K_segmented, num_segments*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    H2Opus_Real** d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    H2Opus_Real** d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    TLR_Matrix matrix;
    cudaErr = cudaMalloc((void**) &matrix.blockRanks, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    int k_sum = 0;

    cudaEvent_t startGenerateInputMatrix, stopGenerateInputMatrix;
    cudaEventCreate(&startGenerateInputMatrix);
    cudaEventCreate(&stopGenerateInputMatrix);
    cudaEventRecord(startGenerateInputMatrix);
    
    for(unsigned int segment = 0; segment < num_segments; ++segment){
        dim3 m_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 m_numBlocks(1, num_segments);

        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(config.n, num_segments, maxSegmentSize, config.dim, d_values_in, d_input_matrix_segmented, d_dataset, d_offsets_sort, segment, matrix.diagonal);
        cudaDeviceSynchronize();

        cudaMemcpy(input_matrix_segmented, d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

        #if PRINT_OUTPUT
        printf("input matrix\n");
        for(unsigned int i=0; i<num_segments*maxSegmentSize*maxSegmentSize; ++i){
            printf("%lf ", input_matrix_segmented[i]);
        }
        #endif

        cudaEvent_t startSVD, stopSVD;
        cudaEventCreate(&startSVD);
        cudaEventCreate(&stopSVD);
        cudaEventRecord(startSVD);
        SVD(config.n, num_segments, input_matrix_segmented, maxSegmentSize, h_S, h_U, h_V);
        cudaEventRecord(stopSVD);
        cudaEventSynchronize(stopSVD);
        cudaEventDestroy(startSVD);
        cudaEventDestroy(stopSVD);
        cudaDeviceSynchronize();

        cudaMemcpy(d_S, h_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U, h_U, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

        #if 0
        printSigmas(h_S, num_segments, maxSegmentSize, config.bucket_size, config.n, segment);  
        #endif

        numThreadsPerBlock = maxSegmentSize; //TODO: make sure that bucket_size is less than 1024
        numBlocks = num_segments;
        calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (maxSegmentSize, matrix.blockRanks + segment*num_segments, d_S, config.epsilon);
        cudaDeviceSynchronize();

        #if 0
        unsigned int *h_k = (unsigned int *)malloc(num_segments*sizeof(unsigned int));
        cudaMemcpy(h_k, matrix.blockRanks + segment*num_segments, num_segments*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printKs(h_k, num_segments, maxSegmentSize, config.bucket_size, config.n, segment, config.epsilon);
        free(h_k);
        #endif

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* totalMem = (int*)malloc(sizeof(int));
        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();

        cudaErr = cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaFree(d_totalMem);

        H2Opus_Real* d_U_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        H2Opus_Real* d_V_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

        dim3 d_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 d_numBlocks(1, num_segments);
        tileMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (config.n, num_segments, maxSegmentSize, d_S, d_U, d_V, d_U_tiled_segmented, d_V_tiled_segmented, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, segment);
        cudaDeviceSynchronize();

        #if EXPAND_MATRIX
        H2Opus_Real* d_expMatrix;
        cudaErr = cudaMalloc((void**) &d_expMatrix, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

        expandMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (num_segments, maxSegmentSize, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, d_U_tiled_segmented, d_V_tiled_segmented, d_expMatrix);
        cudaDeviceSynchronize();

        // #if 0
        // printExpM<<<1, 1>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix_segmented);
        // cudaDeviceSynchronize();
        // #endif

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

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments*maxSegmentSize*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
        calcError<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix_segmented, d_error, d_tmp);
        cudaDeviceSynchronize();
        cudaMemcpy(error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaFree(d_error);
        cudaFree(d_tmp);
        printf("error: %lf\n", sqrt(*error)/sqrt(*tmp));
        free(tmp);
        free(error);
        cudaFree(d_expMatrix);
        #endif

        cudaErr = cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaMemcpy(d_U_tiled_temp[segment], d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_V_tiled_temp[segment], d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

        k_sum += (*totalMem);

        free(totalMem);
        cudaFree(d_U_tiled_segmented);
        cudaFree(d_V_tiled_segmented);
    }
    printf("total mem %d\n", k_sum);
    timer_arr[5] =k_sum;

    cudaEventRecord(stopGenerateInputMatrix);
    cudaEventSynchronize(stopGenerateInputMatrix);
    float GenMatrix_time = 0;
    cudaEventElapsedTime(&GenMatrix_time, startGenerateInputMatrix, stopGenerateInputMatrix);
    timer_arr[6] = GenMatrix_time;
    cudaEventDestroy(startGenerateInputMatrix);
    cudaEventDestroy(stopGenerateInputMatrix);

    free(h_S);
    free(h_U);
    free(h_V);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_scan_K_segmented);
    free(input_matrix_segmented);
    cudaFree(d_values_in);
    cudaFree(d_offsets_sort);
    cudaFree(d_dataset);
    cudaFree(d_input_matrix_segmented);

    cudaErr = cudaMalloc((void**) &matrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix.blockOffsets, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    unsigned int* h_scan_K = (unsigned int*)malloc(num_segments*num_segments*sizeof(unsigned int));
    cudaErr = cudaMemcpy(h_scan_K, matrix.blockOffsets, num_segments*num_segments*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    for(unsigned int segment = 0; segment < num_segments-1; ++segment){
        cudaErr = cudaMemcpy(&matrix.U[h_scan_K[num_segments*segment]*maxSegmentSize], d_U_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(int), cudaMemcpyDeviceToDevice);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMemcpy(&matrix.V[h_scan_K[num_segments*segment]*maxSegmentSize], d_V_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(int), cudaMemcpyDeviceToDevice);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    }

    printf("copied data\n");
    for(unsigned int segment = 0; segment < num_segments; ++segment){
        cudaFree(d_U_tiled_temp[segment]);
        cudaFree(d_V_tiled_temp[segment]);
    }
    free(h_scan_K);
    free(d_U_tiled_temp);
    free(d_V_tiled_temp);
    
    // TODO: cudafree d_U_tiled_temp and d_V_tiled_temp

    H2Opus_Real* d_buffer_vector;
    H2Opus_Real* d_input_vector;
    H2Opus_Real* d_output_vector;
    H2Opus_Real* d_output_vector_org;
    cudaErr = cudaMalloc((void**) &d_buffer_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_buffer_vector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_input_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_input_vector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_output_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_output_vector_org, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024;
    numBlocks = (num_segments*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
    fillVector<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_input_vector, d_output_vector, d_output_vector_org);
    cudaErr = cudaGetLastError();
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error fillVector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaDeviceSynchronize();
    printf("filled vector\n");

    numThreadsPerBlock = 2*upper_power_of_two(maxSegmentSize);
    numBlocks = (num_segments+1)/2;

    cudaEvent_t startGEMV, stopGEMV;
    cudaEventCreate(&startGEMV);
    cudaEventCreate(&stopGEMV);
    cudaEventRecord(startGEMV);
    GEMV<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, matrix.blockRanks, matrix.blockOffsets, matrix.U, matrix.V, matrix.diagonal, d_input_vector, d_output_vector, d_buffer_vector);
    cudaErr = cudaGetLastError();
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaDeviceSynchronize();
    cudaEventRecord(stopGEMV);
    cudaEventSynchronize(stopGEMV);
    float GEMV_time = 0;
    cudaEventElapsedTime(&GEMV_time, startGEMV, stopGEMV);
    cudaEventDestroy(startGEMV);
    cudaEventDestroy(stopGEMV);

    // dense matrix * vector
    #if 1
    // generate n*n dense matrix
    H2Opus_Real* d_denseMatrix;
    cudaErr = cudaMalloc((void**) &d_denseMatrix, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_vector;
    cudaErr = cudaMalloc((void**) &d_vector, config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024; //TODO: make sure that bucket_size is less than 1024   
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
    float DenseGEMV_time = 0;
    cudaEventElapsedTime(&DenseGEMV_time, startDenseGEMV, stopDenseGEMV);
    printf("DenseGEMV time: %f\n", DenseGEMV_time);
    timer_arr[8] = DenseGEMV_time;
    cudaEventDestroy(startDenseGEMV);
    cudaEventDestroy(stopDenseGEMV);

    cublasDestroy(handle);
    cudaFree(d_denseMatrix);
    cudaFree(d_vector);
    #endif

    printf("GEMV time: %f\n", GEMV_time);
    timer_arr[7] = GEMV_time;

    cudaFree(d_buffer_vector);
    cudaFree(d_input_vector);
    cudaFree(d_output_vector);
    cudaFree(d_output_vector_org);

    // H2Opus_Real* d_U_tiled_2;
    // H2Opus_Real* d_V_tiled_2;
    // int* d_scan_K_2;
    // int* d_K_2;
    TLR_Matrix matrix2;
    cudaErr = cudaMalloc((void**) &matrix2.blockRanks, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.diagonal, maxSegmentSize*maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.blockOffsets, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    cudaMemcpy(matrix2.blockRanks, matrix.blockRanks, num_segments*num_segments*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.blockOffsets, matrix.blockOffsets, num_segments*num_segments*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.U, matrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.V, matrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.diagonal, matrix.diagonal, maxSegmentSize*maxSegmentSize*num_segments*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

    H2Opus_Real* d_U_tiled_output;
    H2Opus_Real* d_V_tiled_output;
    int* d_scan_K_output;
    TLR_Matrix matrix_gemm_output;

    cudaErr = cudaMalloc((void**) &matrix_gemm_output.blockRanks, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_U_tiled_output, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_V_tiled_output, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_scan_K_output, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    h_S = (H2Opus_Real *)malloc(maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));

    d_S;
    d_U;
    d_V;
    cudaErr = cudaMalloc((void**) &d_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_U, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_V, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    H2Opus_Real* d_gemm_matrix_segmented;
    H2Opus_Real* gemm_matrix_segmented = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    cudaErr = cudaMalloc((void**) &d_gemm_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    d_scan_K_segmented;
    cudaErr = cudaMalloc((void**) &d_scan_K_segmented, num_segments*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    k_sum=0;

    dim3 mm_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
    dim3 mm_numBlocks(1, num_segments);
    numThreadsPerBlock = maxSegmentSize; //TODO: make sure that bucket_size is less than 1024   
    numBlocks = num_segments;

    cudaEvent_t startGEMM, stopGEMM;
    cudaEventCreate(&startGEMM);
    cudaEventCreate(&stopGEMM);
    cudaEventRecord(startGEMM);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        GEMM<<<mm_numBlocks, mm_numThreadsPerBlock, 2*config.bucket_size*config.bucket_size*sizeof(H2Opus_Real)>>>(num_segments, maxSegmentSize, matrix.U, matrix.V, matrix.diagonal, matrix.blockRanks, matrix.blockOffsets, matrix2.U, matrix2.V, matrix2.diagonal, matrix2.blockRanks, matrix2.blockOffsets, d_gemm_matrix_segmented, segment, config.bucket_size);
        cudaDeviceSynchronize();
        cudaMemcpy(gemm_matrix_segmented, d_gemm_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

        // SVD on gemm_matrix
        SVD(config.n, num_segments, gemm_matrix_segmented, maxSegmentSize, h_S, h_U, h_V);
        cudaDeviceSynchronize();

        cudaMemcpy(d_S, h_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U, h_U, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

        // calcmemneeded and truncation on gemm matrix
        calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (maxSegmentSize, matrix_gemm_output.blockRanks + segment*num_segments, d_S, config.epsilon);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* totalMem = (int*)malloc(sizeof(int));
        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();

        cudaErr = cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaFree(d_totalMem);

        H2Opus_Real* d_U_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        H2Opus_Real* d_V_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

        dim3 d_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 d_numBlocks(1, num_segments);
        tileMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (config.n, num_segments, maxSegmentSize, d_S, d_U, d_V, d_U_tiled_segmented, d_V_tiled_segmented, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, segment);
        cudaDeviceSynchronize();

        cudaErr = cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaMemcpy(d_U_tiled_temp[segment], d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_V_tiled_temp[segment], d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

        k_sum += (*totalMem);

        free(totalMem);
        cudaFree(d_U_tiled_segmented);
        cudaFree(d_V_tiled_segmented);
        // save output in a double pointer array
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
    // TODO: copy from double pointer array to a single pointer array

    #if 1
    // generate n*n dense matrix
    H2Opus_Real* d_denseMatrix1;
    cudaErr = cudaMalloc((void**) &d_denseMatrix1, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_denseMatrix2;
    cudaErr = cudaMalloc((void**) &d_denseMatrix2, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_denseMatrix3;
    cudaErr = cudaMalloc((void**) &d_denseMatrix3, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024; //TODO: make sure that bucket_size is less than 1024   
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

    free(gemm_matrix_segmented);
    cudaFree(d_scan_K_segmented);
    cudaFree(matrix_gemm_output.blockRanks);
    cudaFree(d_gemm_matrix_segmented);
    
    cudaFreeMatrix(matrix);
    cudaFreeMatrix(matrix2);

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