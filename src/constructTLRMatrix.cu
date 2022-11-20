
#include <assert.h>

#include "constructTLRMatrix.cuh"
#include "cublas_v2.h"
#include "helperKernels.cuh"
#include "kblas.h"
#include "batch_rand.h"

__global__ void getTotalMem(unsigned int* totalMem, int* K, int* scan_K, int num_segments){
    *totalMem = scan_K[num_segments - 1] + K[num_segments - 1];
}

__global__ void fillARAArrays(int batchCount, int maxSegmentSize, int* d_rows_batch, int* d_cols_batch, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchCount){
        d_rows_batch[i] = maxSegmentSize;
        d_cols_batch[i] = maxSegmentSize;
        d_ldm_batch[i] = maxSegmentSize;
        d_lda_batch[i] = maxSegmentSize;
        d_ldb_batch[i] = maxSegmentSize;
    }
}

__global__ void copyTiles(int batchCount, int maxSegmentSize, int* d_ranks, unsigned int* d_scan_k, H2Opus_Real* d_U_tiled_segmented, H2Opus_Real* d_A, H2Opus_Real* d_V_tiled_segmented, H2Opus_Real* d_B, unsigned int maxRank){
    if(threadIdx.x < d_ranks[blockIdx.x]) {
        unsigned int scanRanks = d_scan_k[blockIdx.x] - d_ranks[blockIdx.x];
        for(unsigned int i = 0; i < maxSegmentSize; ++i) {
            d_U_tiled_segmented[scanRanks*maxSegmentSize + threadIdx.x*maxSegmentSize + i] = d_A[blockIdx.x*maxSegmentSize*maxRank + threadIdx.x*maxSegmentSize + i];
            d_V_tiled_segmented[scanRanks*maxSegmentSize + threadIdx.x*maxSegmentSize + i] = d_B[blockIdx.x*maxSegmentSize*maxRank + threadIdx.x*maxSegmentSize + i];
        }
    }
}

__global__ void copyRanks(int num_segments, int maxSegmentSize, int* from_ranks, int* to_ranks){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_segments*(num_segments-1)){
        int row = i%(num_segments-1);
        int col = i/(num_segments-1);
        int diff = (row>=col) ? 1 : 0;
        to_ranks[i + col + diff] = from_ranks[i];
    }
    if(i < num_segments){
        to_ranks[i*num_segments + i] = 0;
    }
}

__device__ H2Opus_Real getCorrelationLength(int dim){
    return dim == 3 ? 0.2 : 0.1;
}

__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud){
    assert(col<n);
    assert(row<n);

    H2Opus_Real diff = 0;
    H2Opus_Real x, y;
    for (int d = 0; d < dim; ++d){
        x = pointCloud[d*n + col];
        y = pointCloud[d*n + row];
        diff += (x - y)*(x - y);
    }

    H2Opus_Real dist = sqrt(diff);
    return exp(-dist/getCorrelationLength(dim));
}

__global__ void generateDenseBlockColumn(unsigned int numberOfInputPoints, unsigned int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* matrix, H2Opus_Real* pointCloud, KDTree kDTree, int columnIndex, H2Opus_Real* diagonal) {
    for(unsigned int i = 0; i < (maxSegmentSize/blockDim.x); ++i) {
        for(unsigned int j = 0; j < (maxSegmentSize/blockDim.x); ++j) {
            unsigned int row = blockIdx.y*maxSegmentSize + i*blockDim.x + threadIdx.y;
            unsigned int col = columnIndex*maxSegmentSize + j*blockDim.x + threadIdx.x;

            if(blockIdx.y == columnIndex) {
                diagonal[columnIndex*maxSegmentSize*maxSegmentSize + j*maxSegmentSize*blockDim.x + threadIdx.x*maxSegmentSize + i*blockDim.y + threadIdx.y] = interaction(numberOfInputPoints, dimensionOfInputPoints, kDTree.segmentIndices[kDTree.segmentOffsets[columnIndex] + blockDim.x*j + threadIdx.x], kDTree.segmentIndices[kDTree.segmentOffsets[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
            }
            else {
                unsigned int diff = (blockIdx.y > columnIndex) ? 1 : 0;
                unsigned int matrixIndex = blockIdx.y*maxSegmentSize*maxSegmentSize - diff*maxSegmentSize*maxSegmentSize + j*blockDim.x*maxSegmentSize + threadIdx.x*maxSegmentSize + i*blockDim.y + threadIdx.y;
                int xDim = kDTree.segmentOffsets[columnIndex + 1] - kDTree.segmentOffsets[columnIndex];
                int yDim = kDTree.segmentOffsets[blockIdx.y + 1] - kDTree.segmentOffsets[blockIdx.y];

                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    if(col == row) {
                        matrix[matrixIndex] = 1;
                    }
                    else {
                        matrix[matrixIndex] = 0;
                    }
                }
                else {
                    matrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, kDTree.segmentIndices[kDTree.segmentOffsets[columnIndex] + blockDim.x*j + threadIdx.x], kDTree.segmentIndices[kDTree.segmentOffsets[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                }
            }
        }
    }
}

unsigned int createColumnMajorLRMatrix(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int dimensionOfInputPoints, TLR_Matrix &matrix, KDTree kDTree, H2Opus_Real* &d_dataset, float tolerance, int ARA_R) {
    unsigned int maxRank = kDTree.segmentSize/2;
    printf("maxRanks: %d\n", maxRank);

    int *d_rowsBatch, *d_colsBatch, *d_ranks;
    int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_MPtrs, **d_APtrs, **d_BPtrs;
    cudaMalloc((void**) &d_rowsBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_colsBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_ranks, (kDTree.numSegments - 1)*kDTree.numSegments*sizeof(int));
    cudaMalloc((void**) &d_LDMBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_LDABatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_LDBBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_A, (kDTree.numSegments - 1)*kDTree.segmentSize*maxRank*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_B, (kDTree.numSegments - 1)*kDTree.segmentSize*maxRank*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_MPtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_APtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_BPtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*));

    int numThreadsPerBlock = 1024;
    int numBlocks = (kDTree.numSegments - 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments - 1, kDTree.segmentSize, d_rowsBatch, d_colsBatch, d_LDMBatch, d_LDABatch, d_LDBBatch);

    kblasHandle_t kblasHandle;
    kblasRandState_t randState;
    kblasCreate(&kblasHandle);
    kblasInitRandState(kblasHandle, &randState, 1 << 15, 0);
    kblasEnableMagma(kblasHandle);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, bucketSize, kDTree.numSegments - 1);
    kblasAllocateWorkspace(kblasHandle);

    float ARATotalTime = 0;
    unsigned int rankSum = 0;
    unsigned int* totalMem = (unsigned int*)malloc(sizeof(unsigned int));
    unsigned int* d_totalMem;
    cudaMalloc((void**) &d_totalMem, sizeof(unsigned int));

    H2Opus_Real* d_inputMatrixSegmented;
    unsigned int* d_scanRanksSegmented;
    cudaMalloc((void**) &d_inputMatrixSegmented, kDTree.segmentSize*kDTree.segmentSize*kDTree.numSegments*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_scanRanksSegmented, (kDTree.numSegments - 1)*sizeof(unsigned int));
    cudaMalloc((void**) &matrix.blockRanks, kDTree.numSegments*kDTree.numSegments*sizeof(int));
    cudaMalloc((void**) &matrix.diagonal, kDTree.numSegments*kDTree.segmentSize*kDTree.segmentSize*sizeof(H2Opus_Real));
    H2Opus_Real **d_UTiledTemp = (H2Opus_Real**)malloc(kDTree.numSegments*sizeof(H2Opus_Real*));
    H2Opus_Real **d_VTiledTemp = (H2Opus_Real**)malloc(kDTree.numSegments*sizeof(H2Opus_Real*));

    dim3 m_numThreadsPerBlock(min(32, (int)kDTree.segmentSize), min(32, (int)kDTree.segmentSize));
    dim3 m_numBlocks(1, kDTree.numSegments);

    for(unsigned int segment = 0; segment < kDTree.numSegments; ++segment) {
        generateDenseBlockColumn <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, kDTree.segmentSize, dimensionOfInputPoints, d_inputMatrixSegmented, d_dataset, kDTree, segment, matrix.diagonal);
        cudaDeviceSynchronize();

        generateArrayOfPointersT<H2Opus_Real>(d_inputMatrixSegmented, d_MPtrs, kDTree.segmentSize*kDTree.segmentSize, kDTree.numSegments - 1, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, kDTree.segmentSize*maxRank, kDTree.numSegments - 1, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, kDTree.segmentSize*maxRank, kDTree.numSegments - 1, 0);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        int kblas_ara_return = kblas_ara_batch(
            kblasHandle, d_rowsBatch, d_colsBatch, d_MPtrs, d_LDMBatch, 
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks + segment*(kDTree.numSegments - 1),
            tolerance, kDTree.segmentSize, kDTree.segmentSize, maxRank, 16, ARA_R, randState, 0, kDTree.numSegments - 1
        );
        assert(kblas_ara_return == 1);
        cudaDeviceSynchronize();

        void* d_tempStorage = NULL;
        size_t tempStorageBytes = 0;
        cub::DeviceScan::InclusiveSum(d_tempStorage, tempStorageBytes, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceScan::InclusiveSum(d_tempStorage, tempStorageBytes, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaDeviceSynchronize();
        cudaFree(d_tempStorage);

        cudaMemcpy(totalMem, d_scanRanksSegmented + kDTree.numSegments - 2, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaMalloc((void**) &d_UTiledTemp[segment], kDTree.segmentSize*(*totalMem)*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_VTiledTemp[segment], kDTree.segmentSize*(*totalMem)*sizeof(H2Opus_Real));

        // TODO: optimize thread allocation here OR replace with cudaMemcpys
        int numThreadsPerBlock = kDTree.segmentSize;
        int numBlocks = kDTree.numSegments - 1;
        copyTiles <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments - 1, kDTree.segmentSize, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, d_UTiledTemp[segment], d_A, d_VTiledTemp[segment], d_B, maxRank);
        rankSum += (*totalMem);
    }

    kblasDestroy(&kblasHandle);
    kblasDestroyRandState(randState);

    free(totalMem);
    cudaFree(d_totalMem);
    cudaFree(d_inputMatrixSegmented);
    cudaFree(d_scanRanksSegmented);
    cudaFree(d_rowsBatch);
    cudaFree(d_colsBatch);
    cudaFree(d_LDMBatch);
    cudaFree(d_LDABatch);
    cudaFree(d_LDBBatch);
    cudaFree(d_MPtrs);
    cudaFree(d_APtrs);
    cudaFree(d_BPtrs);
    cudaFree(d_A);
    cudaFree(d_B);

    cudaMalloc((void**) &matrix.U, rankSum*kDTree.segmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrix.V, rankSum*kDTree.segmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments*sizeof(int));

    numThreadsPerBlock = 1024;
    numBlocks = ((kDTree.numSegments - 1)*kDTree.numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    // TODO: no need for this. Instead, replace d_ranks with matrix.blockRanks
    copyRanks <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments, kDTree.segmentSize, d_ranks, matrix.blockRanks);
    cudaDeviceSynchronize();
    cudaFree(d_ranks);

    void *d_tempStorage = NULL;
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, matrix.blockRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments);
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, matrix.blockRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments);
    cudaDeviceSynchronize();
    cudaFree(d_tempStorage);

    int* h_scanRanks = (int*)malloc(kDTree.numSegments*kDTree.numSegments*sizeof(int));
    cudaMemcpy(h_scanRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int segment = 0; segment < kDTree.numSegments - 1; ++segment) {
        cudaMemcpy(&matrix.U[h_scanRanks[kDTree.numSegments*segment]*kDTree.segmentSize], d_UTiledTemp[segment], (h_scanRanks[kDTree.numSegments*(segment + 1)] - h_scanRanks[kDTree.numSegments*segment])*kDTree.segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&matrix.V[h_scanRanks[kDTree.numSegments*segment]*kDTree.segmentSize], d_VTiledTemp[segment], (h_scanRanks[kDTree.numSegments*(segment + 1)] - h_scanRanks[kDTree.numSegments*segment])*kDTree.segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(&matrix.U[h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)]*kDTree.segmentSize], d_UTiledTemp[kDTree.numSegments - 1], (rankSum - h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*kDTree.segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&matrix.V[h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)]*kDTree.segmentSize], d_VTiledTemp[kDTree.numSegments - 1], (rankSum - h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*kDTree.segmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    free(h_scanRanks);

    for(unsigned int segment = 0; segment < kDTree.numSegments; ++segment) {
        cudaFree(d_UTiledTemp[segment]);
        cudaFree(d_VTiledTemp[segment]);
    }
    free(d_UTiledTemp);
    free(d_VTiledTemp);
    
    return rankSum;
}
