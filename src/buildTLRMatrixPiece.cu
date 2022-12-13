
#include "buildTLRMatrixPiece.cuh"
#include "helperKernels.cuh"
#include "kDTree.cuh"
#include "magma_auxiliary.h"

bool isPieceDiagonal(unsigned int pieceMortonIndex);

template <class T>
void buildTLRMatrixPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_pointDataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tolerance) {

        magma_init();

        matrix->tileSize = kdtree.leafSize;
        unsigned int numTilesInPieceRow = (upperPowerOfTwo(kdtree.N)/numPiecesInAxis)/matrix->tileSize;
        unsigned int numTilesInPieceCol = isPieceDiagonal(pieceMortonIndex) ? numTilesInPieceRow : numTilesInPieceRow - 1;

        int maxRank = matrix->tileSize>>1;
        int *d_rowsBatch, *d_colsBatch, *d_ranksOutput;
        int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
        T *d_UOutput, *d_VOutput;
        T **d_MPtrs, **d_UOutputPtrs, **d_VOutputPtrs;

        cudaMalloc((void**) &d_rowsBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_ranksOutput, numTilesInPieceCol*kdtree.numLeaves*sizeof(int));
        cudaMalloc((void**) &d_LDMBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_LDABatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_LDBBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_UOutput, numTilesInPieceCol*matrix->tileSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_VOutput, numTilesInPieceCol*matrix->tileSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_MPtrs, numTilesInPieceCol*sizeof(T*));
        cudaMalloc((void**) &d_UOutputPtrs, numTilesInPieceCol*sizeof(T*));
        cudaMalloc((void**) &d_VOutputPtrs, numTilesInPieceCol*sizeof(T*));

        kblasHandle_t kblasHandle;
        kblasRandState_t randState;
        kblasCreate(&kblasHandle);
        kblasInitRandState(kblasHandle, &randState, 1 << 15, 0);
        kblasEnableMagma(kblasHandle);
        kblas_ara_batch_wsquery<T>(kblasHandle, matrix->tileSize, numTilesInPieceCol);
        kblasAllocateWorkspace(kblasHandle);

        magma_finalize();
}

bool isPieceDiagonal(unsigned int pieceMortonIndex) {
    uint32_t pieceRow, pieceCol;
    morton2columnMajor((uint32_t)pieceMortonIndex, pieceCol, pieceRow);
    return (pieceCol == pieceRow);
}

template void buildTLRMatrixPiece<H2Opus_Real>(
    TLR_Matrix *matrix,
    KDTree kdtree,
    H2Opus_Real* d_dataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    H2Opus_Real tolerance);
