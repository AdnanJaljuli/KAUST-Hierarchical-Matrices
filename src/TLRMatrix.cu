
#include <assert.h>
#include "TLRMatrix.cuh"

void freeMatrix(TLR_Matrix matrix){
    cudaFree(matrix.blockRanks);
    cudaFree(matrix.blockOffsets);
    cudaFree(matrix.U);
    cudaFree(matrix.V);
    cudaFree(matrix.diagonal);
}
