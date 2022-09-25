
// TODO: make all header files independent
#include "config.h"
#include "counters.h"
#include "createLRMatrix.cuh"
#include "helperFunctions.cuh"
#include "hierarchicalMatrixFunctions.cuh"
#include "kblas.h"
#include "kdtreeConstruction.cuh"
#include "tlr_example.h"
#include "TLR_Matrix.h"

#include <algorithm>
#include <assert.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <typeinfo>
#include <utility>
using namespace std;

// TODO: make EXPAND_MATRIX a config argument
#define EXPAND_MATRIX 1

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
    printf("n: %d\n", config.numberOfInputPoints);
    printf("bucket size: %d\n", config.bucketSize);
    printf("epsilon: %f\n", config.lowestLevelTolerance);
    printf("dimensionOfInputPoints: %d\n", config.dimensionOfInputPoints);
    printf("num counters: %d\n", NUM_COUNTERS);

    // TODO: fix counters (use enum, not macros; see vertex cover code for how we define counters in an extensible way)
    Counters counters;
    initCounters(&counters);

    H2Opus_Real* d_dataset;
    gpuErrchk(cudaMalloc((void**) &d_dataset, config.numberOfInputPoints*config.dimensionOfInputPoints*sizeof(H2Opus_Real)));
    generateDataset(config.numberOfInputPoints, config.dimensionOfInputPoints, d_dataset);

    uint64_t maxNumSegments = (config.numberOfInputPoints + config.bucketSize - 1)/config.bucketSize; // TODO: use camel case everywhere
    printf("max num segments: %d\n", maxNumSegments);
    uint64_t numSegments;
    int  *d_valuesIn;
    int  *d_offsetsSort;
    cudaMalloc((void**) &d_valuesIn, config.numberOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_offsetsSort, (maxNumSegments + 1)*sizeof(int));
    // TODO: move the frees for the two mallocs above to the end of the main function
    createKDTree(config.numberOfInputPoints, config.dimensionOfInputPoints, config.bucketSize, numSegments, config.divMethod, d_valuesIn, d_offsetsSort, d_dataset, maxNumSegments);

    uint64_t maxSegmentSize = config.bucketSize;
    printf("max segment size: %lu\n", maxSegmentSize);
    printf("num segments: %lu\n", numSegments);
    const int ARA_R = 10;
    int max_rows = maxSegmentSize;
    int max_cols = maxSegmentSize;
    int max_rank = max_cols;
    TLR_Matrix matrix;
    matrix.type = COLUMN_MAJOR;
    H2Opus_Real* d_denseMatrix;
    uint64_t kSum = createColumnMajorLRMatrix(config.numberOfInputPoints, numSegments, maxSegmentSize, config.bucketSize, config.dimensionOfInputPoints, matrix, d_denseMatrix, d_valuesIn, d_offsetsSort, d_dataset, config.lowestLevelTolerance, ARA_R, max_rows, max_cols, max_rank);
    gpuErrchk(cudaPeekAtLastError());

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, matrix, d_denseMatrix);
    #endif

    TLR_Matrix mortonMatrix;
    mortonMatrix.type = MORTON;
    ConvertColumnMajorToMorton(numSegments, maxSegmentSize, kSum, matrix, mortonMatrix); // TODO: Do not capitalize the first letter of function names
    
    matrix.cudaFreeMatrix();

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, mortonMatrix, d_denseMatrix);
    #endif


    const int numLevels = __builtin_ctz(config.numberOfInputPoints/config.bucketSize) + 1;
    printf("numLevels: %d\n", numLevels);
    int** HMatrixExistingRanks = (int**)malloc((numLevels - 1)*sizeof(int*));
    int** HMatrixExistingTiles = (int**)malloc((numLevels - 1)*sizeof(int*));
    genereateHierarchicalMatrix(config.numberOfInputPoints, config.bucketSize, numSegments, maxSegmentSize, numLevels, mortonMatrix, HMatrixExistingRanks, HMatrixExistingTiles);

    mortonMatrix.cudaFreeMatrix();
    gpuErrchk(cudaPeekAtLastError());

    cudaEvent_t stopCode;
    cudaEventCreate(&stopCode);
    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float codeTime = 0;
    cudaEventElapsedTime(&codeTime, startCode, stopCode);
    // counters[11] = codeTime;
    printf("total time: %f\n", codeTime);

    #if 0
    printCountersInFile(config, counters);
    #endif
    // Clean up
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    cudaFree(d_dataset);
    #if 0
    free(counters);
    #endif

    return 0;
}

