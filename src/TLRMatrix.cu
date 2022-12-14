
#include "TLRMatrix.cuh"

void freeMatrix(TLR_Matrix *matrix){
    matrix->d_tileOffsets.clear();
    matrix->d_U.clear();
    matrix->d_V.clear();
}
