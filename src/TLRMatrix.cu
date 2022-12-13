
#include "TLRMatrix.cuh"

void freeMatrix(TLR_Matrix *matrix){
    matrix->d_blockOffsets.clear();
    matrix->d_U.clear();
    matrix->d_V.clear();
    matrix->d_diagonal.clear();
}
