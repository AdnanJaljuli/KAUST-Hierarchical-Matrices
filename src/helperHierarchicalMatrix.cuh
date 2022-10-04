
#ifndef __HELPERHIERARCHICALMATRIX_H__
#define __HELPERHIERARCHICALMATRIX_H__

static __device__ unsigned int tileInQuarter(unsigned int row, unsigned int col, unsigned int dimension) {
    if(row >= dimension && col >= dimension) {
        return 3;
    }
    else if(row >= dimension) {
        return 2;
    }
    else if(col >= dimension) {
        return 1;
    }
    else {
        return 0;
    }
}

static __device__ unsigned int numPrecedingDiagonalTiles(unsigned int row, unsigned int col, int numSegments) {
    unsigned int answer = 0;
    while(true) {
        numSegments /= 2;
        unsigned int quarter = tileInQuarter(row, col, numSegments);
        if(quarter != 0) {
            answer += numSegments;
        }
        if(quarter == 1 || quarter == 2) {
            break;
        }
        if(quarter == 3) {
            row -= numSegments;
            col -= numSegments;
        }
    }
    return answer;
}

static __global__ void fillFirstLevel(int numSegments, int* existingTiles, int* existingRanks, int* mortonOrderedMatrixRanks) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numSegments*numSegments) {
        unsigned int row = i%numSegments;
        unsigned int col = i/numSegments;
        if(col != row) {
            int mortonOrderedIndex = IndextoMOIndex_h(numSegments, i);
            unsigned int tmp = numPrecedingDiagonalTiles(row, col, numSegments);
            unsigned int writeIndex = mortonOrderedIndex - tmp;
            printf("i: %d.   precedingdiag: %d.   write index: %d\n", i, tmp, writeIndex);
            existingTiles[writeIndex] = mortonOrderedIndex;
            existingRanks[writeIndex] = mortonOrderedMatrixRanks[i];
        }
    }
}

static __global__ void getNewLevelCount(int num_ops, int* d_new_bit_vector, int* d_new_bit_vector_scan, int* d_newLevelCount){
    d_newLevelCount[0] = d_new_bit_vector_scan[num_ops - 1] + d_new_bit_vector[num_ops - 1];
}

static __global__ void copyTilesToNewLevel(int num_ops, int* bit_vector, TLR_Matrix mortonOrderedMatrix, H2Opus_Real* d_A, H2Opus_Real* d_B, int* new_ranks, int* old_active_tiles, int row, int col){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_ops){
        if(bit_vector[i] == 1){
            // TODO: fix the address to where the function will copy
            // TODO: use multiple streams
            cudaMemcpyAsync(&mortonOrderedMatrix.U[mortonOrderedMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_A[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(&mortonOrderedMatrix.V[mortonOrderedMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_B[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
        }
    }
}

static __global__ void calcNumOps(int num_existing_tiles, int* num_ops, int* availableTiles){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_existing_tiles){
        if(availableTiles[i]%4 == 0){
            bool flag = true;
            for(int j=1; j<4; ++j){
                if(availableTiles[i+j] != availableTiles[i]+j){
                    flag = false;
                    break;
                }
            }
            if(flag){
                atomicAdd(num_ops, 1);
            }
        }
    }
}

static __global__ void expandHMatrixLevel(int num_ops, int maxRows, int maxCols, H2Opus_Real* d_A, H2Opus_Real* d_B, int* d_ranks, H2Opus_Real* expandedMatrix){
    int col = threadIdx.x + blockIdx.x*(maxCols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(maxRows/2);
    int block = blockIdx.y/2;
    H2Opus_Real sum = 0;

    for(unsigned int i=0; i<d_ranks[block]; ++i){
        sum += d_A[maxRows*maxCols*block + i*maxRows + row]*d_B[maxRows*maxCols*block + i*maxRows + col];
    }
    expandedMatrix[block*maxRows*maxCols + col*maxRows + row] = sum;
}

static __global__ void errorInHMatrix(int num_segments, int maxSegmentSize, int num_ops, int maxRows, int maxCols, H2Opus_Real* expandedMatrix, H2Opus_Real* d_denseMatrix, int* activeTiles, H2Opus_Real* d_error, H2Opus_Real* d_tmp){
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