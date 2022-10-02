
#include "config.h"
#include "counters.h"
#include "createLRMatrix.cuh"
#include "expandMatrixHelpers.cuh"
#include "helperFunctions.cuh"
#include "hierarchicalMatrixFunctions.cuh"
#include "kblas.h"
#include "kDTreeConstruction.cuh"
#include "TLRMatrix.h"

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

int main(int argc, char *argv[]) {

    cudaDeviceSynchronize();

    Config config = parseArgs(argc, argv);
    printArgs(config);

    #if USE_COUNTERS
    Counters counters;
    initCounters(&counters);
    startTime(TOTAL_TIME, &counters);
    #endif

    // Generate the points
    H2Opus_Real* d_pointCloud;
    gpuErrchk(cudaMalloc((void**) &d_pointCloud, config.numberOfInputPoints*config.dimensionOfInputPoints*sizeof(H2Opus_Real)));
    generateDataset(config.numberOfInputPoints, config.dimensionOfInputPoints, d_pointCloud);

    // Build the KD-tree
    // TODO: Combine into a struct that represents the KD-tree
    uint64_t numSegments = (config.numberOfInputPoints + config.bucketSize - 1)/config.bucketSize;
    int  *d_valuesIn; // TODO: rename to something more representative
    int  *d_offsetsSort; // TODO: rename to something more representative
    cudaMalloc((void**) &d_valuesIn, config.numberOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_offsetsSort, (numSegments + 1)*sizeof(int));
    createKDTree(config.numberOfInputPoints, config.dimensionOfInputPoints, numSegments, config.bucketSize, d_valuesIn, d_offsetsSort, d_pointCloud);

    // Build the TLR matrix
    uint64_t maxSegmentSize = config.bucketSize;
    printf("max segment size: %lu\n", maxSegmentSize);
    printf("num segments: %lu\n", numSegments);
    const int ARA_R = 10;
    int max_rows = maxSegmentSize;
    int max_cols = maxSegmentSize;
    int max_rank = max_cols;
    TLR_Matrix matrix;
    matrix.ordering = COLUMN_MAJOR;
    uint64_t rankSum = createColumnMajorLRMatrix(config.numberOfInputPoints, numSegments, maxSegmentSize, config.bucketSize, config.dimensionOfInputPoints, matrix, d_valuesIn, d_offsetsSort, d_pointCloud, config.lowestLevelTolerance, ARA_R, max_rows, max_cols, max_rank);

    #if EXPAND_MATRIX
    // TODO: assert that this doesn't exceed memory limit
    H2Opus_Real* d_denseMatrix;
    cudaMalloc((void**) &d_denseMatrix, numSegments*numSegments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    generateDenseMatrix(config.numberOfInputPoints, numSegments, maxSegmentSize, config.dimensionOfInputPoints, d_denseMatrix, d_valuesIn, d_offsetsSort, d_pointCloud);
    #endif

    cudaFree(d_pointCloud);
    cudaFree(d_valuesIn);
    cudaFree(d_offsetsSort);
    gpuErrchk(cudaPeekAtLastError());

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, matrix, d_denseMatrix);
    #endif

    // Convert TLR matrix to morton order
    TLR_Matrix mortonMatrix;
    mortonMatrix.ordering = MORTON;
    ConvertColumnMajorToMorton(numSegments, maxSegmentSize, rankSum, matrix, mortonMatrix); // TODO: Do not capitalize the first letter of function names    
    cudaFreeMatrix(matrix);

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(numSegments, maxSegmentSize, mortonMatrix, d_denseMatrix);
    #endif

    // Build hierarchical matrix
    // TODO: move declarations not used later inside the function
    #if 0
    const int numLevels = __builtin_ctz(config.numberOfInputPoints/config.bucketSize) + 1;
    printf("numLevels: %d\n", numLevels);
    int** HMatrixExistingRanks = (int**)malloc((numLevels - 1)*sizeof(int*));
    int** HMatrixExistingTiles = (int**)malloc((numLevels - 1)*sizeof(int*));
    genereateHierarchicalMatrix(config.numberOfInputPoints, config.bucketSize, numSegments, maxSegmentSize, numLevels, mortonMatrix, HMatrixExistingRanks, HMatrixExistingTiles);
    #endif

    cudaFreeMatrix(mortonMatrix);
    gpuErrchk(cudaPeekAtLastError());
    #if EXPAND_MATRIX
    cudaFree(d_denseMatrix);
    #endif

    #if USE_COUNTERS
    endTime(TOTAL_TIME, &counters);
    printCountersInFile(config, &counters);
    #endif

    return 0;
}