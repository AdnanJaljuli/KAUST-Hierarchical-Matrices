
#include "TLRMatrix.cuh"

void freeTLRPiece(TLR_Matrix *matrix){
    cudaFree(matrix->d_tileOffsets);
    matrix->d_U.clear();
    matrix->d_V.clear();
}
