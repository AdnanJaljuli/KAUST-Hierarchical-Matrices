
#include "constructHMatrixFromStruct.cuh"
#include <cub/cub.cuh>
#include "cublas_v2.h"
#include "helperKernels.cuh"
#include "HMatrixHelpers.cuh"
#include "HMatrix.cuh"
#include "kblas.h"
#include "batch_rand.h"
#include "TLRMatrix.cuh"

// TODO: break this code into smaller pieces
void generateHMatrixFromStruct(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R, float tolerance, HMatrix hierarchicalMatrix, WeakAdmissibility WAStruct, unsigned int *maxRanks) {

    kblasHandle_t kblasHandle;
    kblasRandState_t randState;
    kblasCreate(&kblasHandle);
    kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
    kblasEnableMagma(kblasHandle);

    // TODO: get rid of maxCols
    int maxRows;
    int maxCols;
    int *d_rowsBatch, *d_colsBatch, *d_ranks;
    int *d_LDABatch, *d_LDBBatch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_APtrs, **d_BPtrs;
    // TODO: allocate memory outside the loop
    // TODO: use multiple streams

    for(unsigned int level = WAStruct.numLevels - 2; level > 0; --level) {
        int batchSize = WAStruct.numTiles[level - 1];
        if(batchSize == 0) {
            continue;
        }
        int batchUnitSize = 1 << (WAStruct.numLevels - (level + 1));

        // preprocessing
        int* d_tileIndices;
        cudaMalloc((void**) &d_tileIndices, batchSize*sizeof(int));
        cudaMemcpy(d_tileIndices, WAStruct.tileIndices[level], batchSize*sizeof(int), cudaMemcpyHostToDevice);

        // pointers to level tiles in U and V
        LevelTilePtrs tilePtrs;
        allocateTilePtrs(batchSize, batchUnitSize, segmentSize, level, d_tileIndices, tilePtrs, mortonOrderedMatrix);

        // scan the ranks array
        int *d_scanRanks;
        int **d_scanRanksPtrs;
        cudaMalloc((void**) &d_scanRanks, batchSize*batchUnitSize*batchUnitSize*sizeof(int));
        cudaMalloc((void**) &d_scanRanksPtrs, batchSize*sizeof(int*));
        generateScanRanks(batchSize, batchUnitSize, mortonOrderedMatrix.blockRanks, d_scanRanks, d_scanRanksPtrs, WAStruct.tileIndices[level - 1]);

        tolerance *= 2;
        maxRows = batchUnitSize*bucketSize;
        maxCols = maxRows;
        cudaMalloc((void**) &d_ranks, batchSize*sizeof(int)); // TODO: allocate and free outside loop and reuse memory pools across levels
        cudaMalloc((void**) &d_rowsBatch, batchSize*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, batchSize*sizeof(int));
        cudaMalloc((void**) &d_LDABatch, batchSize*sizeof(int));
        cudaMalloc((void**) &d_LDBBatch, batchSize*sizeof(int));
        cudaMalloc((void**) &d_APtrs, batchSize*sizeof(H2Opus_Real*));
        cudaMalloc((void**) &d_BPtrs, batchSize*sizeof(H2Opus_Real*));
        cudaMalloc((void**) &d_A, static_cast<uint64_t>(batchSize)*maxRows*maxRanks[WAStruct.numLevels - level - 2]*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_B, static_cast<uint64_t>(batchSize)*maxRows*maxRanks[WAStruct.numLevels - level - 2]*sizeof(H2Opus_Real));

        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillLRARAArrays <<< numBlocks, numThreadsPerBlock >>> (batchSize, maxRows, d_rowsBatch, d_colsBatch, d_LDABatch, d_LDBBatch);

        generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, maxRows*maxRanks[WAStruct.numLevels - level - 2], batchSize, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, maxRows*maxRanks[WAStruct.numLevels - level - 2], batchSize, 0);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, batchSize);
        kblasAllocateWorkspace(kblasHandle);

        int LRARAReturnVal = lr_kblas_ara_batch(kblasHandle, segmentSize, batchUnitSize, d_rowsBatch, d_colsBatch, tilePtrs.U, tilePtrs.V, d_scanRanksPtrs,
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks,
            tolerance, maxRows, maxCols, maxRanks[WAStruct.numLevels - level - 2], 16, ARA_R, randState, 0, batchSize
        );
        assert(LRARAReturnVal == 1);

        // allocate HMatrix level
        allocateAndCopyToHMatrixLevel(hierarchicalMatrix.levels[level - 1], d_ranks, WAStruct, level, d_A, d_B, maxRows, maxRanks[WAStruct.numLevels - level - 2]);

        // free memory
        freeLevelTilePtrs(tilePtrs);
        cudaFree(d_tileIndices);
        cudaFree(d_scanRanks);
        cudaFree(d_scanRanksPtrs);
        cudaFree(d_ranks);
        cudaFree(d_rowsBatch);
        cudaFree(d_colsBatch);
        cudaFree(d_LDABatch);
        cudaFree(d_LDBBatch);
        cudaFree(d_APtrs);
        cudaFree(d_BPtrs);
        cudaFree(d_A);
        cudaFree(d_B);
    }

    kblasDestroy(&kblasHandle);
    kblasDestroyRandState(randState);
}

__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize, int batchSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        d_scanRanksPtrs[i] = &d_scanRanks[i*batchUnitSize*batchUnitSize];
    }
}

__global__ void fillLRARAArrays(int batchSize, int maxRows, int* d_rowsBatch, int* d_colsBatch, int* d_LDABatch, int* d_LDBBatch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize){
        d_rowsBatch[i] = maxRows;
        d_colsBatch[i] = maxRows;
        d_LDABatch[i] = maxRows;
        d_LDBBatch[i] = maxRows;
    }
}

void generateScanRanks(int batchSize, int batchUnitSize, int *ranks, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices) {
    // TODO: we already have a scanRanks array of all the ranks in the MOMatrix. Use that one instead of this
    for(unsigned int batch = 0; batch < batchSize; ++batch) {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks + levelTileIndices[batch]*batchUnitSize*batchUnitSize, scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks + levelTileIndices[batch]*batchUnitSize*batchUnitSize, scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        cudaFree(d_temp_storage);
    }

    // fillScanRanksPtrs
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (scanRanksPtrs, scanRanks, batchUnitSize, batchSize);
}