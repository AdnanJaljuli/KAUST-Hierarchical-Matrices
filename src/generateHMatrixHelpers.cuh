
#ifndef __HELPERS_HIERARCHICALMATRIX_H__
#define __HELPERS_HIERARCHICALMATRIX_H__

__global__ void fillBatchedPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize) {
    d_UPtrs[0] = &mortonOrderedMatrix.U[mortonOrderedMatrix.blockOffsets[4]*segmentSize];
    d_VPtrs[0] = &mortonOrderedMatrix.V[mortonOrderedMatrix.blockOffsets[4]*segmentSize];
    d_UPtrs[1] = &mortonOrderedMatrix.U[mortonOrderedMatrix.blockOffsets[8]*segmentSize];
    d_VPtrs[1] = &mortonOrderedMatrix.V[mortonOrderedMatrix.blockOffsets[8]*segmentSize];
}

__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize) {
    d_scanRanksPtrs[0] = &d_scanRanks[0];
    d_scanRanksPtrs[1] = &d_scanRanks[batchUnitSize*batchUnitSize];
}

__global__ void fillLRARAArrays(int batchSize, int maxRows, int maxCols, int* d_rowsBatch, int* d_colsBatch, int* d_LDABatch, int* d_LDBBatch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize){
        d_rowsBatch[i] = maxRows;
        d_colsBatch[i] = maxRows;
        d_LDABatch[i] = maxRows;
        d_LDBBatch[i] = maxRows;
    }
}

__global__ void fillTempScanRanksPtr(int* d_ranksStartingPtr, int* blockRanks) {
    d_ranksStartingPtr[0] = blockRanks[64];
}

__global__ void fillBatchSegments(int *batchSegments, int batchUnitSize, int batchSize) {
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < batchSize*batchUnitSize*batchUnitSize) {
        batchSegments[i] = i/(batchUnitSize*batchUnitSize);
    }
}

static __global__ void fillExistingTilesBitVector(int numSegments, uint64_t *existingTileBits) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numSegments*numSegments) {
        unsigned int row = i%numSegments;
        unsigned int col = i/numSegments;
        int mortonOrderedIndex = IndextoMOIndex_h(numSegments, i);
        if(row != col) {
            unsigned int pos = mortonOrderedIndex%(sizeof(uint64_t)*8);
            unsigned int sub = mortonOrderedIndex/(sizeof(uint64_t)*8);
            atomicOr((unsigned long long*)&existingTileBits[sub], 1ULL<<(sizeof(uint64_t)*8 - 1 - pos));
        }
    }
}

static __global__ void fillBitVectorPopCnt(int bitVectorSize, uint64_t* existingTileBits, int *popCExistingTileBits) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < bitVectorSize) {
        popCExistingTileBits[i] = __popcll(existingTileBits[i]);
    }
}

static __global__ void fillInitialHMatrixLevel_kernel(int numSegments, uint64_t* existingTileBits, int* popCExistingTilesBitScan, int* existingTiles, int* existingRanks, int *mortonOrderedMatrixRanks) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numSegments*numSegments) {
        unsigned int row = i%numSegments;
        unsigned int col = i/numSegments;
        if(row != col) {
            int mortonOrderedIndex = IndextoMOIndex_h(numSegments, i);
            unsigned int pos = mortonOrderedIndex%(sizeof(uint64_t)*8);
            unsigned int sub = mortonOrderedIndex/(sizeof(uint64_t)*8);
            unsigned int onesToLeft = pos - __popcll(existingTileBits[sub]>>(sizeof(uint64_t)*8 - pos));
            unsigned int index = popCExistingTilesBitScan[sub] + pos - onesToLeft;
            assert(index < numSegments*(numSegments - 1));
            existingRanks[index] = mortonOrderedMatrixRanks[mortonOrderedIndex];
            existingTiles[index] = mortonOrderedIndex;
        }
    }
}

static void fillInitialHMatrixLevel(int numSegments, int* existingTiles, int* existingRanks, int *mortonOrderedMatrixRanks) {
    // fill bit vector
    uint64_t* d_existingTilesBitVector;
    unsigned int bitVectorSize = (numSegments*numSegments + sizeof(uint64_t)*8 - 1)/(sizeof(uint64_t)*8);
    printf("bit vector size: %d\n", bitVectorSize);
    cudaMalloc((void**) &d_existingTilesBitVector, bitVectorSize*sizeof(uint64_t));
    cudaMemset(d_existingTilesBitVector, 0, bitVectorSize*sizeof(uint64_t));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numSegments*numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillExistingTilesBitVector <<< numBlocks, numThreadsPerBlock >>> (numSegments, d_existingTilesBitVector);

    // fill pop count array
    int* d_popCExistingTilesBitVector;
    cudaMalloc((void**) &d_popCExistingTilesBitVector, bitVectorSize*sizeof(int));
    numThreadsPerBlock = 1024;
    numBlocks = (bitVectorSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillBitVectorPopCnt <<< numBlocks, numThreadsPerBlock >>> (bitVectorSize, d_existingTilesBitVector, d_popCExistingTilesBitVector);

    // scan over pop count array
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int* d_popCExistingTileBitVectorScan;
    cudaMalloc((void**) &d_popCExistingTileBitVectorScan, bitVectorSize*sizeof(int));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_popCExistingTilesBitVector, d_popCExistingTileBitVectorScan, bitVectorSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_popCExistingTilesBitVector, d_popCExistingTileBitVectorScan, bitVectorSize);
    cudaFree(d_temp_storage);

    // launch a kernel to figure out where each thread will write
    numThreadsPerBlock = 1024;
    numBlocks = (numSegments*numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillInitialHMatrixLevel_kernel <<< numBlocks, numThreadsPerBlock >>> (numSegments, d_existingTilesBitVector, d_popCExistingTileBitVectorScan, existingTiles, existingRanks, mortonOrderedMatrixRanks);
    cudaDeviceSynchronize();
}

static __global__ void calcNumOps(int numExistingTiles, int* numOps, int* availableTiles) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numExistingTiles) {
        if(availableTiles[i]%4 == 0) {
            bool flag = true;
            for(int j = 1; j < 4; ++j) {
                if(availableTiles[i + j] != availableTiles[i] + j) {
                    flag = false;
                    break;
                }
            }
            if(flag) {
                atomicAdd(numOps, 1);
            }
        }
    }
}

static __global__ void getNewLevelCount(int numOps, int* d_new_bit_vector, int* d_new_bit_vector_scan, int* d_newLevelCount){
    d_newLevelCount[0] = d_new_bit_vector_scan[numOps - 1] + d_new_bit_vector[numOps - 1];
}

static __global__ void copyTilesToNewLevel(int numOps, int* bit_vector, TLR_Matrix mortonOrderedMatrix, H2Opus_Real* d_A, H2Opus_Real* d_B, int* new_ranks, int* old_active_tiles, int row, int col){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < numOps){
        if(bit_vector[i] == 1){
            // TODO: fix the address to where the function will copy
            // TODO: use multiple streams
            cudaMemcpyAsync(&mortonOrderedMatrix.U[mortonOrderedMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_A[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(&mortonOrderedMatrix.V[mortonOrderedMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_B[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
        }
    }
}

static __global__ void expandHMatrixLevel(int numOps, int maxRows, int maxCols, H2Opus_Real* d_A, H2Opus_Real* d_B, int* d_ranks, H2Opus_Real* expandedMatrix){
    int col = threadIdx.x + blockIdx.x*(maxCols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(maxRows/2);
    int block = blockIdx.y/2;
    H2Opus_Real sum = 0;

    for(unsigned int i=0; i<d_ranks[block]; ++i){
        sum += d_A[maxRows*maxCols*block + i*maxRows + row]*d_B[maxRows*maxCols*block + i*maxRows + col];
    }
    expandedMatrix[block*maxRows*maxCols + col*maxRows + row] = sum;
}

static __global__ void errorInHMatrix(int num_segments, int maxSegmentSize, int numOps, int maxRows, int maxCols, H2Opus_Real* expandedMatrix, H2Opus_Real* d_denseMatrix, int* activeTiles, H2Opus_Real* d_error, H2Opus_Real* d_tmp){
    int col = threadIdx.x + blockIdx.x*(maxCols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(maxRows/2);
    int block = blockIdx.y/2;

    int MOIndex = activeTiles[block*4]/4;
    int i = morton1(MOIndex);
    int j = morton1(MOIndex >> 1);

    H2Opus_Real x = d_denseMatrix[(col + i*maxCols)*num_segments*maxSegmentSize + j*maxRows + row];
    H2Opus_Real y = expandedMatrix[block*maxRows*maxCols + col*maxRows + row];
    atomicAdd(d_tmp, x*x);
    atomicAdd(d_error, (x-y)*(x-y));
}

#endif