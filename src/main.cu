
// TODO: make all header files independent

// TODO: alphabetical order
#include "tlr_example.h"
#include "TLR_Matrix.h"
#include "helperFunctions.cuh"
#include "config.h"
#include "kdtreeConstruction.cuh"
#include "createLRMatrix.cuh"
#include "hierarchicalMatrix.cuh"

// TODO: alphabetical order
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
using namespace std;

// TODO: make EXPAND_MATRIX a config argument
#define EXPAND_MATRIX 1
#define BLOCK_SIZE 32

int main(int argc, char *argv[]) {

    cudaDeviceSynchronize();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice name: %s\n\n", prop.name);

    cudaEvent_t startCode;
    cudaEventCreate(&startCode);
    cudaEventRecord(startCode);

    Config config = parseArgs(argc, argv);
    // TODO: move printfs below to their own function printArgs defined inside of config.h
    printf("n: %d\n", config.n);
    printf("bucket size: %d\n", config.bucket_size);
    printf("epsilon: %f\n", config.tol);
    printf("dim: %d\n", config.dim);

    // TODO: fix counters (use enum, not macros; see vertex cover code for how we define counters in an extensible way)
    float* counters = (float*)malloc(NUM_COUNTERS*sizeof(float));
    counters[0] = (float)config.n;
    counters[1] = (float)config.bucket_size;
    counters[2] = (float)config.dim;
    counters[3] = (float)config.tol;

    H2Opus_Real* d_dataset;
    gpuErrchk(cudaMalloc((void**) &d_dataset, config.n*config.dim*sizeof(H2Opus_Real)));
    generateDataset(config.n, config.dim);

    uint64_t max_num_segments = (config.n + config.bucket_size - 1)/config.bucket_size; // TODO: use camel case everywhere
    printf("max num segments: %d\n", max_num_segments);
    uint64_t numSegments;
    int  *d_values_in;
    int  *d_offsets_sort;
    cudaMalloc((void**) &d_values_in, n*sizeof(int));
    cudaMalloc((void**) &d_offsets_sort, (max_num_segments + 1)*sizeof(int));
    // TODO: mode the frees for the two mallocs above to the end of the main function
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
    uint64_t k_sum = createColumnMajorLRMatrix(config.n, numSegments, maxSegmentSize, config.bucket_size, config.dim, matrix, d_denseMatrix, d_values_in, d_offsets_sort, d_dataset, config.tol, ARA_R, max_rows, max_cols, max_rank);
    gpuErrchk(cudaPeekAtLastError());

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, matrix, d_denseMatrix);
    #endif

    TLR_Matrix mortonMatrix;
    mortonMatrix.type = MORTON;
    ConvertColumnMajorToMorton(numSegments, maxSegmentSize, k_sum, matrix, mortonMatrix); // TODO: Do not capitalize the first letter of function names
    
    matrix.cudaFreeMatrix();

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, mortonMatrix, d_denseMatrix);
    #endif

    return 0; // XXX: XXX XXX XXX

    const int numLevels = __builtin_ctz(config.n/config.bucket_size) + 1;
    printf("numLevels: %d\n", numLevels);
    int** HMatrixExistingRanks = (int**)malloc((numLevels - 1)*sizeof(int*));
    int** HMatrixExistingTiles = (int**)malloc((numLevels - 1)*sizeof(int*));
    genereateHierarchicalMatrix(config.n, config.bucket_size, numSegments, maxSegmentSize, numLevels, mortonMatrix, HMatrixExistingRanks, HMatrixExistingTiles);

    mortonMatrix.cudaFreeMatrix();
    gpuErrchk(cudaPeekAtLastError());

    cudaEvent_t stopCode;
    cudaEventCreate(&stopCode);
    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float Code_time = 0; //  TODO: do not capitalize first letter of identifiers
    cudaEventElapsedTime(&Code_time, startCode, stopCode);
    counters[11] = Code_time;
    printf("total time: %f\n", Code_time);

    printCountersInFile(counters);

    // Clean up
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    cudaFree(d_dataset);
    free(counters);

}

