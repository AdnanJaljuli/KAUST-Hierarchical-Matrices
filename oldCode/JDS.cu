
#include "JDS.cuh"
#include "HMatrix.cuh"

void buildMatrixJDS(MatrixJDS *JDS, HMatrixStructure HMatrixStruct) {

    JDS->levels = (HMatrixLevelJDS*)malloc((HMatrixStruct.numLevels - 1)*sizeof(HMatrixLevelJDS));
    for(unsigned int level = 1; level < HMatrixStruct.numLevels; ++level) {
        int numRows = 1<<level;
        JDS->levels->rowIndices = (int*)malloc(numRows*sizeof(int));
        for(unsigned int i = 0; i < numRows; ++i) {
            JDS->levels->rowIndices[i] = i;
        }
        
    }
}