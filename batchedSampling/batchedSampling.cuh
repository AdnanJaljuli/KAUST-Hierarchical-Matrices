#ifndef __BATCHED_SAMPLING__
#define __BATCHED_SAMPLING__

// this function takes as input:
// - an array of pointers to where each batch unit starts
// - a scan ranks array
// - an array of sampling vectors
// - sampling vectors' width
// - batch size
// - unit size
// - tile size

static __global__ void batchedSampling(int tileSize, int batchSize, int unitSize, double** U, double** V, int* scanRanks, double* samplingVectors, int widthSamplingVector) {
    
}

#endif