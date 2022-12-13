
#include "buildTLRMatrixPiece.cuh"
#include "helperKernels.cuh"
#include "kDTree.cuh"

bool isPieceDiagonal(unsigned int pieceMortonIndex);

template <class T>
void buildTLRMatrixPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_pointDataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tolerance) {

        bool isDiagonal = isPieceDiagonal(pieceMortonIndex);
        unsigned int numTilesInPieceRow = (upperPowerOfTwo(kdtree.N)/numPiecesInAxis)/kdtree.leafSize;
        unsigned int numTilesInPieceCol = isDiagonal ? numTilesInPieceRow : numTilesInPieceRow - 1;

        int maxRank = kdtree.leafSize>>1;
        int *d_rowsBatch, *d_colsBatch, *d_ranksOutput;
        int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
        T *d_UOutput, *d_VOutput;
        T **d_MPtrs, **d_UOutputPtrs, **d_VOutputPtrs;

        cudaMalloc((void**) &d_rowsBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_ranksOutput, numTilesInPieceCol*kdtree.numSegments*sizeof(int));
        cudaMalloc((void**) &d_LDMBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_LDABatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_LDBBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_UOutput, numTilesInPieceCol*kdtree.leafSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_VOutput, numTilesInPieceCol*kdtree.leafSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_MPtrs, numTilesInPieceCol*sizeof(T*));
        cudaMalloc((void**) &d_UOutputPtrs, numTilesInPieceCol*sizeof(T*));
        cudaMalloc((void**) &d_VOutputPtrs, numTilesInPieceCol*sizeof(T*));
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
