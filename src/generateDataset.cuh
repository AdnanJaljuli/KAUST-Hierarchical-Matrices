
#ifndef ___GENERATE_DATASET__
#define ___GENERATE_DATASET__

typedef double H2Opus_Real;
__global__ void generateDataset_kernel(int numberOfInputPoints, int dimensionOfInputPoints, H2Opus_Real* pointCloud);
void generateDataset(int numberOfInputPoints, int dimensionOfInputPoints, H2Opus_Real* d_pointCloud);

#endif
