#include "TLR_Matrix.h"
#include "TLR_Matrix.cu"

void cudaFreeMatrix(TLR_Matrix matrix){
    cudaFree(matrix.blockRanks);
    cudaFree(matrix.blockOffsets);
    cudaFree(matrix.U);
    cudaFree(matrix.V);
    // cudaFree(matrix.diagonal);
}