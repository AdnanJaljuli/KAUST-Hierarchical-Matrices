
#include "TLRMatrix.cuh"

void freeTLRMatrix(TLR_Matrix *matrix){
    matrix->d_tileOffsets.clear();
    matrix->d_U.clear();
    matrix->d_V.clear();
}
