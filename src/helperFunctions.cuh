
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include "TLRMatrix.cuh"
#include "HMatrix.cuh"
#include "helperKernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

template <class T>
void convertColumnMajorToMorton(TLR_Matrix matrix, TLR_Matrix *mortonMatrix);

void printMatrixStructure(HMatrixStructure HMatrixStruct);
void printPointCloud(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real *d_pointCloud);
void printKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, DIVISION_METHOD divMethod, unsigned int leafSize, KDTree tree, H2Opus_Real* d_pointCloud);

#endif