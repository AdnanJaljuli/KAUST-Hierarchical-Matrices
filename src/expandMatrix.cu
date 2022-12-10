
#include <assert.h>
#include <stdio.h>

#include "cublas_v2.h"
#include "helperKernels.cuh"
#include "HMatrix.cuh"
#include "TLRMatrixHelpers.cuh"

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
    printf("error in LR matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_expandedMatrix);
}

__global__ void expandHMatrix(HMatrixLevel matrixLevel, H2Opus_Real* output, int tileSize) {
    unsigned int batch = blockIdx.x;
    unsigned int col = blockIdx.y*blockDim.y + threadIdx.x;
    unsigned int row = blockIdx.z*blockDim.z + threadIdx.y;

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
    unsigned int batch = blockIdx.x;
    unsigned int col = blockIdx.y*blockDim.y + threadIdx.x;
    unsigned int row = blockIdx.z*blockDim.z + threadIdx.y;

    if(col < batchUnitSize*32 && row < batchUnitSize*32) {
        uint32_t tileRow, tileColumn;
        mortonToCM((uint32_t)tileIndices[batch], tileColumn, tileRow);

        double x = denseMatrix[(tileColumn*batchUnitSize*32 + col)*numberOfInputPoints + tileRow*32*batchUnitSize + row];
        double y = output[batch*batchUnitSize*32*batchUnitSize*32 + col*batchUnitSize*32 + row];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
    }
}

void checkErrorInHMatrixLevel(int numberOfInputPoints, int batchSize, int batchUnitSize, int bucketSize, HMatrix matrix, H2Opus_Real *denseMatrix, int level) {
    // expand H matrix level
    H2Opus_Real *d_expandedMatrix;
    cudaMalloc((void**) &d_expandedMatrix, batchSize*batchUnitSize*bucketSize*batchUnitSize*bucketSize*sizeof(H2Opus_Real));
    dim3 m_numThreadsPerBlock(32, 32);
    dim3 m_numBlocks(batchSize, batchUnitSize, batchUnitSize);
    expandHMatrix <<< m_numBlocks, m_numThreadsPerBlock >>> (matrix.levels[level - 1], d_expandedMatrix, batchUnitSize*bucketSize);

    int* d_tileIndices;
    cudaMalloc((void**) &d_tileIndices, batchSize*sizeof(int));
    cudaMemcpy(d_tileIndices, matrix.matrixStructure.tileIndices[level - 1], batchSize*sizeof(int), cudaMemcpyHostToDevice);
    
    // compare expanded H matrix level with dense matrix
    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));
    errorInHMatrix <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, denseMatrix, d_expandedMatrix, d_tileIndices, batchSize, batchUnitSize, d_error, d_tmp);
    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_expandedMatrix);
    cudaFree(d_tileIndices);
    printf("error in hierarchical matrix level %d is: %lf\n", level, sqrt(h_error)/sqrt(h_tmp));
}

void checkErrorInHMatrix(int numberOfInputPoints, int bucketSize, HMatrix hierarchicalMatrix, H2Opus_Real* d_denseMatrix) {
    
    for(unsigned int level = hierarchicalMatrix.matrixStructure.numLevels - 1; level > 0; --level) {
        int batchSize = hierarchicalMatrix.matrixStructure.numTiles[level - 1];
        if(batchSize == 0) {
            continue;
        }
        int batchUnitSize = 1 << (hierarchicalMatrix.matrixStructure.numLevels - (level + 1));
        checkErrorInHMatrixLevel(numberOfInputPoints, batchSize, batchUnitSize, bucketSize, hierarchicalMatrix, d_denseMatrix, level);
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

void generateDenseMatrix(int numberOfInputPoints, int numSegments, int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* d_denseMatrix, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* &d_dataset) {
    dim3 m_numThreadsPerBlock(min(32, (int)maxSegmentSize), min(32, (int)maxSegmentSize));
    dim3 m_numBlocks(numSegments, numSegments);
    generateDenseMatrix_kernel <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, numSegments, maxSegmentSize, dimensionOfInputPoints, d_denseMatrix, d_valuesIn, d_offsetsSort, d_dataset);
}

__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  double alpha,
  double *A,
  int lda,
  double *B,
  int ldb,
  double beta,
  double *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    double accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}


void checkErrorInHmatrixVecMult(unsigned int numberOfInputPoints, unsigned int vectorWidth, int numSegments, H2Opus_Real *d_denseMatrix, H2Opus_Real *d_inputVectors, H2Opus_Real *d_resultVectors) {

    H2Opus_Real *d_denseXVec;
    cudaMalloc((void**) &d_denseXVec, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    dim3 block(16, 16);
    dim3 grid(
        (numberOfInputPoints + block.x - 1) / block.x,
        (vectorWidth + block.y - 1) / block.y
    );

  ReferenceGemm_kernel<<< grid, block >>>(numberOfInputPoints, vectorWidth, numberOfInputPoints, 1, d_denseMatrix, numberOfInputPoints, d_inputVectors, numberOfInputPoints, 0, d_denseXVec, numberOfInputPoints);

    H2Opus_Real *h_denseXVec = (H2Opus_Real*)malloc(numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    H2Opus_Real *h_resultVectors = (H2Opus_Real*)malloc(numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    cudaMemcpy(h_resultVectors, d_resultVectors, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_denseXVec, d_denseXVec, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    char filename[100] = "results/hmatvec.txt";
    FILE *output_file = fopen(filename, "a");
    H2Opus_Real tmp = 0;
    H2Opus_Real error = 0;
    for(unsigned int i = 0; i < numberOfInputPoints; ++i) {
		for(unsigned int j = 0; j < vectorWidth; ++j) {
            fprintf(output_file, "%d  %d    %lf           %lf\n", i, j, h_resultVectors[j*numberOfInputPoints + i], h_denseXVec[j*numberOfInputPoints + i]);
            tmp += h_denseXVec[j*numberOfInputPoints + i]*h_denseXVec[j*numberOfInputPoints + i];
            error += (h_denseXVec[j*numberOfInputPoints + i]-h_resultVectors[j*numberOfInputPoints + i])*(h_denseXVec[j*numberOfInputPoints + i]-h_resultVectors[j*numberOfInputPoints + i]);
        }
        fprintf(output_file, "\n");
    }
    printf("error in hmatvec: %lf\n", sqrt(error)/sqrt(tmp));
    fprintf(output_file, "\n");
    fclose(output_file);
}