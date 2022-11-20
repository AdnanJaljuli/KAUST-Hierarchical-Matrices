
#include <assert.h>
#include <stdio.h>

#include "helperKernels.cuh"
#include "HMatrix.cuh"
#include "constructTLRMatrix.cuh"

__global__ void expandLRMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* expandedMatrix, TLR_Matrix matrix) {
    if(blockIdx.x == blockIdx.y) {
        expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = matrix.diagonal[blockIdx.x*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    }
    else{
        unsigned int index;
        if(matrix.ordering == MORTON) {
            index = CMIndextoMOIndex(num_segments, blockIdx.x*num_segments + blockIdx.y);
        }
        else if(matrix.ordering == COLUMN_MAJOR) {
            index = blockIdx.x*num_segments + blockIdx.y;
        }

        H2Opus_Real sum = 0;
        for(unsigned int i=0; i<matrix.blockRanks[index]; ++i) {
            sum += matrix.U[matrix.blockOffsets[index]*maxSegmentSize + i*maxSegmentSize + threadIdx.y]*matrix.V[matrix.blockOffsets[index]*maxSegmentSize + i*maxSegmentSize + threadIdx.x];
        }
        expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = sum;
    }
}

__global__ void errorInLRMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* denseMatrix, H2Opus_Real* expandedMatrix, H2Opus_Real* error, H2Opus_Real* tmp) {
    H2Opus_Real x = denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y];
    H2Opus_Real y = expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    
    atomicAdd(tmp, x*x);
    atomicAdd(error, (x - y)*(x - y));
}

void checkErrorInLRMatrix(unsigned int numSegments, unsigned int maxSegmentSize, TLR_Matrix matrix, H2Opus_Real* d_denseMatrix) {
    H2Opus_Real* d_expandedMatrix;
    cudaMalloc((void**) &d_expandedMatrix, numSegments*maxSegmentSize*numSegments*maxSegmentSize*sizeof(H2Opus_Real));

    dim3 mm_numBlocks(numSegments, numSegments);
    dim3 mm_numThreadsPerBlock(32, 32);
    expandLRMatrix <<< mm_numBlocks, mm_numThreadsPerBlock >>> (numSegments, maxSegmentSize, d_expandedMatrix, matrix);

    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));

    errorInLRMatrix <<< mm_numBlocks, mm_numThreadsPerBlock >>> (numSegments, maxSegmentSize, d_denseMatrix, d_expandedMatrix, d_error, d_tmp);

    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_expandedMatrix);
}

__global__ void expandHMatrix(HMatrixLevel matrixLevel, H2Opus_Real* output, int tileSize) {
    unsigned int batch = blockIdx.z;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(col < tileSize && row < tileSize) {
        H2Opus_Real sum = 0;
        int tileRank = (batch == 0) ? matrixLevel.tileScanRanks[batch] : matrixLevel.tileScanRanks[batch] - matrixLevel.tileScanRanks[batch - 1];
        int tileOffset = (matrixLevel.tileScanRanks[batch] - tileRank)*tileSize;

        for(unsigned int i = 0; i < tileRank; ++i) {
            sum += matrixLevel.U[tileOffset + i*tileSize + row]*matrixLevel.V[tileOffset + i*tileSize + col];
        }
        output[batch*tileSize*tileSize + col*tileSize + row] = sum;
    }
}

__global__ void errorInHMatrix(unsigned int numberOfInputPoints, double* denseMatrix, double* output, int* tileIndices, int batchSize, int batchUnitSize, double* error, double* tmp) {
    unsigned int batch = blockIdx.z;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

    int diff;
    if(batch%2 == 0) {
        diff = 1;
    }
    else {
        diff = -1;
    }

    if(col < batchUnitSize*32 && row < batchUnitSize*32) {
        double x = denseMatrix[(col + batchUnitSize*32*(batch + diff))*numberOfInputPoints + batchUnitSize*32*batch + row];
        double y = output[batch*batchUnitSize*32*batchUnitSize*32 + col*batchUnitSize*32 + row];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
    }
}

void checkErrorInHMatrixLevel(int numberOfInputPoints, int batchSize, int batchUnitSize, int bucketSize, HMatrixLevel matrixLevel, H2Opus_Real *denseMatrix) {
    // expand H matrix level
    H2Opus_Real *d_expandedMatrix;
    cudaMalloc((void**) &d_expandedMatrix, batchSize*batchUnitSize*bucketSize*batchUnitSize*bucketSize*sizeof(H2Opus_Real));
    dim3 m_numThreadsPerBlock(32, 32);
    dim3 m_numBlocks(batchUnitSize, batchUnitSize, batchSize);
    expandHMatrix <<< m_numBlocks, m_numThreadsPerBlock >>> (matrixLevel, d_expandedMatrix, batchUnitSize*bucketSize);
    cudaDeviceSynchronize();

    // compare expanded H matrix level with dense matrix
    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));
    errorInHMatrix <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, denseMatrix, d_expandedMatrix, matrixLevel.tileIndices, batchSize, batchUnitSize, d_error, d_tmp);
    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_expandedMatrix);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
}

void checkErrorInHMatrix(int numberOfInputPoints, int bucketSize, HMatrix hierarchicalMatrix, H2Opus_Real* d_denseMatrix) {
    for(unsigned int level = hierarchicalMatrix.numLevels - 2; level > 0; --level) {
        int batchSize = hierarchicalMatrix.levels[level - 1].numTiles;
        if(batchSize == 0) {
            continue;
        }
        int batchUnitSize = 1 << (hierarchicalMatrix.numLevels - (level + 1));
        checkErrorInHMatrixLevel(numberOfInputPoints, batchSize, batchUnitSize, bucketSize, hierarchicalMatrix.levels[level - 1], d_denseMatrix);
    }
}

__global__ void generateDenseMatrix_kernel(unsigned int numberOfInputPoints, unsigned int numSegments, unsigned int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* denseMatrix, int* indexMap, int* offsetsSort, H2Opus_Real* pointCloud) {
    for(unsigned int i = 0; i < (maxSegmentSize/blockDim.x); ++i){
        for(unsigned int j = 0; j < (maxSegmentSize/blockDim.x); ++j){
            
            unsigned int row = blockIdx.y*maxSegmentSize + i*blockDim.x + threadIdx.y;
            unsigned int col = blockIdx.x*maxSegmentSize + j*blockDim.x + threadIdx.x;

            int xDim = offsetsSort[blockIdx.x + 1] - offsetsSort[blockIdx.x];
            int yDim = offsetsSort[blockIdx.y + 1] - offsetsSort[blockIdx.y];
            unsigned int matrixIndex = col*maxSegmentSize*numSegments + row;

            if(blockIdx.y == blockIdx.x) {
                    denseMatrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, indexMap[offsetsSort[blockIdx.x] + blockDim.x*j + threadIdx.x], indexMap[offsetsSort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
            }
            else {
                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    if (col == row) {
                        denseMatrix[matrixIndex] = 1;
                    }
                    else {
                        denseMatrix[matrixIndex] = 0;
                    }
                }
                else {
                    denseMatrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, indexMap[offsetsSort[blockIdx.x] + blockDim.x*j + threadIdx.x], indexMap[offsetsSort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                }
            }
        }
    }
}

void generateDenseMatrix(int numberOfInputPoints, int numSegments, int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* d_denseMatrix, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* &d_dataset) {
    dim3 m_numThreadsPerBlock(min(32, (int)maxSegmentSize), min(32, (int)maxSegmentSize));
    dim3 m_numBlocks(numSegments, numSegments);
    generateDenseMatrix_kernel <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, numSegments, maxSegmentSize, dimensionOfInputPoints, d_denseMatrix, d_valuesIn, d_offsetsSort, d_dataset);
}