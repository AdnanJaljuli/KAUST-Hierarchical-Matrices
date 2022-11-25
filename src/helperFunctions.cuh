
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include "TLRMatrix.cuh"
#include "helperKernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

void convertColumnMajorToMorton(unsigned int numSegments, unsigned int maxSegmentSize, unsigned int rankSum, TLR_Matrix matrix, TLR_Matrix &mortonMatrix);
__global__ void copyCMRanksToMORanks(int num_segments, int maxSegmentSize, int* matrixRanks, int* mortonMatrixRanks);
void generateRandomVector(unsigned int vectorWidth, unsigned int vetorHeight, H2Opus_Real *vector);

static void printMatrix(int numberOfInputPoints, int numSegments, int segmentSize, TLR_Matrix matrix, int level, int rankSum) {
    // n=512, numLevels=5, level=2, batchSize=4, numTilesInBatch=4
    int* ranks = (int*)malloc(numSegments*numSegments*sizeof(int));
    int* offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(ranks, matrix.blockRanks, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(offsets, matrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    H2Opus_Real* U = (H2Opus_Real*)malloc(rankSum*segmentSize*sizeof(H2Opus_Real));
    H2Opus_Real* V = (H2Opus_Real*)malloc(rankSum*segmentSize*sizeof(H2Opus_Real));
    cudaMemcpy(U, matrix.U, rankSum*segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, matrix.V, rankSum*segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    char fileName[100] = "batchedMatrix.txt";
    FILE *outputFile = fopen(fileName, "w");
    int batchSize = 2;
    int unitSize = 4;
    // int tilesToPrint[64] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239};
    int tilesToPrint[32] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    fprintf(outputFile, "%d %d %d\n", unitSize, segmentSize, batchSize);
    int cnt = 0;
    for(unsigned int i = 0; i < numSegments*numSegments; ++i) {
        if(i == tilesToPrint[cnt]) {
            ++cnt;
            fprintf(outputFile, "%d\n", ranks[i]);
            for(unsigned int j = 0; j < ranks[i]*segmentSize; ++j) {
                fprintf(outputFile, "%lf ", U[offsets[i]*segmentSize + j]);
            }
            fprintf(outputFile, "\n");
            for(unsigned int j = 0; j < ranks[i]*segmentSize; ++j) {
                fprintf(outputFile, "%lf ", V[offsets[i]*segmentSize + j]);
            }
            fprintf(outputFile, "\n");
        }
    }
    fclose(outputFile);
}

static void printDenseMatrix(H2Opus_Real* d_denseMatrix, int size) {
    H2Opus_Real* denseMatrix = (H2Opus_Real*)malloc(size*sizeof(H2Opus_Real));
    cudaMemcpy(denseMatrix, d_denseMatrix, size*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    char fileName[100] = "denseMatrix.txt";
    FILE *outputFile = fopen(fileName, "w");
    for(unsigned int row = 0; row < 256; ++row) {
        for(unsigned int col = 256; col < 512; ++col) {
            fprintf(outputFile, "%lf ", denseMatrix[col*512 + row]);
        }
        fprintf(outputFile, "\n");
    }
}

#endif