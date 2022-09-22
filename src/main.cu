#include "tlr_example.h"
#include "TLR_Matrix.h"
#include "helperFunctions.cuh"
#include "config.h"
#include "kdtreeConstruction.cuh"
#include "createLRMatrix.cuh"
#include "hierarchicalMatrix.cuh"

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

// TODO: make all header files independent
// TODO: make EXPAND_MATRIX a config argument
// TODO: fix timer_arr

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

    H2Opus_Real *d_dataset;
    generateDataset_h(config.n, config.dim, d_dataset);

    uint64_t numSegments = 1;
    uint64_t max_num_segments = (config.n+config.bucket_size-1)/config.bucket_size;
    printf("max num segments: %d\n", max_num_segments);

    int  *d_values_in;
    int  *d_offsets_sort;
    createKDTree(config.n, config.dim, config.bucket_size, numSegments, config.div_method, d_values_in, d_offsets_sort, d_dataset, max_num_segments);

    uint64_t maxSegmentSize = config.bucket_size;
    printf("max segment size: %lu\n", maxSegmentSize);
    printf("num segments: %lu\n", numSegments);

    const int ARA_R = 10;
    int max_rows = maxSegmentSize;
    int max_cols = maxSegmentSize;
    int max_rank = max_cols;

    TLR_Matrix matrix;
    matrix.type = COLUMN_MAJOR;
    H2Opus_Real* d_denseMatrix;

    uint64_t k_sum = createColumnMajorLRMatrix(config.n, numSegments, maxSegmentSize, config.bucket_size, config.dim, matrix, d_denseMatrix, d_values_in, d_offsets_sort, d_dataset, tolerance, ARA_R, max_rows, max_cols, max_rank);
    gpuErrchk(cudaPeekAtLastError());

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, matrix, d_denseMatrix);
    #endif

    TLR_Matrix mortonMatrix;
    mortonMatrix.type = MORTON;
    ConvertColumnMajorToMorton(numSegments, maxSegmentSize, k_sum, matrix, mortonMatrix);
    
    matrix.cudaFreeMatrix();

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, mortonMatrix, d_denseMatrix);
    #endif

    return 0;
    const int numLevels = __builtin_ctz(config.n) - __builtin_ctz(config.bucket_size) + 1;
    printf("numLevels: %d\n", numLevels);
    int** HMatrixExistingRanks = (int**)malloc((numLevels - 1)*sizeof(int*));
    int** HMatrixExistingTiles = (int**)malloc((numLevels - 1)*sizeof(int*));
    genereateHierarchicalMatrix(config.n, config.bucket_size, numSegments, maxSegmentSize, numLevels, mortonMatrix, HMatrixExistingRanks, HMatrixExistingTiles);

    mortonMatrix.cudaFreeMatrix();
    gpuErrchk(cudaPeekAtLastError());

    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float Code_time = 0;
    cudaEventElapsedTime(&Code_time, startCode, stopCode);
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    printf("total time: %f\n", Code_time);
    timer_arr[11] = Code_time;
    printCountersInFile(timer_arr);
    free(timer_arr);
}
