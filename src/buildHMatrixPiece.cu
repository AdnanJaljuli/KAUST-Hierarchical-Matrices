
#include "buildHMatrixPiece.cuh"
#include "buildHMatrix_helpers.cuh"
#include "cublas_v2.h"
#include "HMatrix.cuh"
#include "TLRMatrix.cuh"
#include "precision.h"
#include "kblas.h"
#include "batch_rand.h"

#include <algorithm> 
#include <cub/cub.cuh>
#include <vector>
#include <utility>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

// TODO: add TLRtolerance as a parameter. if eps(TLR) == eps(HMatrix) skip finest level, else: do kblas_ara on finest level to produce a finer tile
template <class T>
void buildHMatrixPiece (
    HMatrix <T> HMat,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel) {

        unsigned int numPiecesInAxis = 1<<pieceLevel;

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
        for(unsigned int tileLevel = HMat.structure.numLevels - 2; tileLevel > 0; --tileLevel) {
            assert(pieceLevel <= tileLevel);
            std::pair<int, int> tilesInPiece = getTilesInPiece(
                HMat.structure.tileIndices[tileLevel],
                tileLevel,
                pieceMortonIndex, pieceLevel);

            if(tilesInPiece.first != 0) {
                unsigned int batchUnitSize = 1 << (HMat.structure.numLevels - (tileLevel + 1));
                int* d_tileIndices;
                cudaMalloc((void**) &d_tileIndices, tilesInPiece.first*sizeof(int));
                cudaMemcpy(
                    d_tileIndices,
                    &HMat.structure.tileIndices[tileLevel - 1][tilesInPiece.second],
                    tilesInPiece.first*sizeof(int),
                    cudaMemcpyHostToDevice);

                LevelTilePtrs <T> tilePtrs;
                allocateTilePtrs <T> (
                    tilesInPiece.first,
                    batchUnitSize,
                    TLRMatrix.tileSize,
                    tileLevel,
                    d_tileIndices,
                    &tilePtrs,
                    TLRMatrix);

                int **d_scanRanksPtrs;
                generateScanRanks(tilesInPiece.first, batchUnitSize, TLRMatrix.d_tileOffsets, d_scanRanksPtrs, d_tileIndices) {

                int ldRanks = TLRMatrix.numTilesInAxis;
                thrust::device_vector<int> d_ldUBatch(tilesInPiece.first, TLRMatrix.numTilesInAxis*TLRMatrix.tileSize);
                thrust::device_vector<int> d_ldVBatch(tilesInPiece.first, TLRMatrix.numTilesInAxis*TLRMatrix.tileSize);

                # if 0

                tolerance *= 2;
                maxRows = batchUnitSize*leafSize;
                maxCols = maxRows;
                cudaMalloc((void**) &d_ranks, batchSize*sizeof(int)); // TODO: allocate and free outside loop and reuse memory pools across levels
                cudaMalloc((void**) &d_rowsBatch, batchSize*sizeof(int));
                cudaMalloc((void**) &d_colsBatch, batchSize*sizeof(int));
                cudaMalloc((void**) &d_LDABatch, batchSize*sizeof(int));
                cudaMalloc((void**) &d_LDBBatch, batchSize*sizeof(int));
                cudaMalloc((void**) &d_APtrs, batchSize*sizeof(H2Opus_Real*));
                cudaMalloc((void**) &d_BPtrs, batchSize*sizeof(H2Opus_Real*));
                cudaMalloc((void**) &d_A, static_cast<uint64_t>(batchSize)*maxRows*maxRanks[HMat.matrixStructure.numLevels - level - 2]*sizeof(H2Opus_Real));
                cudaMalloc((void**) &d_B, static_cast<uint64_t>(batchSize)*maxRows*maxRanks[HMat.matrixStructure.numLevels - level - 2]*sizeof(H2Opus_Real));

                unsigned int numThreadsPerBlock = 1024;
                unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillLRARAArrays <<< numBlocks, numThreadsPerBlock >>> (batchSize, maxRows, d_rowsBatch, d_colsBatch, d_LDABatch, d_LDBBatch);

                generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, maxRows*maxRanks[HMat.matrixStructure.numLevels - level - 2], batchSize, 0);
                generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, maxRows*maxRanks[HMat.matrixStructure.numLevels - level - 2], batchSize, 0);
                kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, batchSize);
                kblasAllocateWorkspace(kblasHandle);

                int LRARAReturnVal = lr_kblas_ara_batch(kblasHandle, segmentSize, batchUnitSize, d_rowsBatch, d_colsBatch, tilePtrs.U, tilePtrs.V, d_scanRanksPtrs,
                    d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks,
                    tolerance, maxRows, maxCols, maxRanks[HMat.matrixStructure.numLevels - level - 2], 16, ARA_R, randState, 0, batchSize
                );
                assert(LRARAReturnVal == 1);

                // allocate HMatrix level
                allocateAndCopyToHMatrixLevel(HMat.levels[level - 1], d_ranks, HMat.matrixStructure, level, d_A, d_B, maxRows, maxRanks[HMat.matrixStructure.numLevels - level - 2]);

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
                #endif
            }
        }
}

template void buildHMatrixPiece <H2Opus_Real> (
    HMatrix <H2Opus_Real> HMat,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize, int batchSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        d_scanRanksPtrs[i] = &d_scanRanks[];
    }
}

void generateScanRanks(int batchSize, int batchUnitSize, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices) {
    // fillScanRanksPtrs
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (scanRanksPtrs, scanRanks, batchUnitSize, batchSize);
}