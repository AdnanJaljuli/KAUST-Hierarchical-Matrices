
#ifndef __GENERATE_HIERARCHICALMATRIX_H__
#define __GENERATE_HIERARCHICALMATRIX_H__

#include "config.h"
#include "counters.h"
#include "generateHMatrixHelpers.cuh"
#include "hierarchicalMatrix.h"
#include "kDTree.h"
#include "TLRMatrix.h"

void generateHMatrixFromStruct(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R, float tolerance) {
    HMatrix hierarchicalMatrix;
    allocateHMatrix(hierarchicalMatrix, segmentSize, numSegments, numberOfInputPoints, bucketSize);
    printf("numHMatrixLevels: %d\n", hierarchicalMatrix.numLevels);
}

#endif