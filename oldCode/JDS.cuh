
#ifndef JDS_H
#define JDS_H

#include "HMatrix.cuh"

struct HMatrixLevelJDS {
    int *rowIndices;
    int *columnIndices;
    int *iterPtr;
};

struct MatrixJDS {
    HMatrixLevelJDS *levels;
};

void buildMatrixJDS(MatrixJDS *JDS, HMatrixStructure HMatrixStruct);

#endif