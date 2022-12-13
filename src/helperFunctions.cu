
#include "helperFunctions.cuh"
#include "HMatrix.cuh"
#include "kDTree.cuh"
#include "kDTreeHelpers.cuh"
#include <curand.h>

void generateMaxRanks(unsigned int numLevels, unsigned int leafSize, unsigned int *maxRanks) {
    for(unsigned int i = 0; i < numLevels - 2; ++i) {
        maxRanks[i] = leafSize*(1 << i);
        if(i > 5) {
            maxRanks[i]/=4;
        }
    }
}

void printMatrixStructure(HMatrixStructure HMatrixStruct) {
    char filename[100] = "results/hmatrixstructure.txt";
    FILE *output_file = fopen(filename, "w");
    fprintf(output_file, "%d\n", HMatrixStruct.numLevels);
    fprintf(output_file, "[ ");

     for(auto i = 0; i <  HMatrixStruct.numLevels - 1;  i++) {
        fprintf(output_file, "[ ");
        int numTiles = HMatrixStruct.numTiles[i];
        for(auto j = 0; j < HMatrixStruct.numTiles[i]; j++) {
            uint32_t x, y;
            morton2CM((uint32_t)(HMatrixStruct.tileIndices[i][j]), x, y);
            fprintf(output_file, "%d, ", x*(1<<i + 1) + y);
        }
        fprintf(output_file, " ], ");
    }
    fprintf(output_file, " ]");
    fclose(output_file);
}

void printPointCloud(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real *d_pointCloud) {
    H2Opus_Real *h_pointCloud = (H2Opus_Real*)malloc(numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real));
    cudaMemcpy(h_pointCloud, d_pointCloud, numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    char filename[100] = "results/pointCloud.txt";
    FILE *output_file = fopen(filename, "w");

    for(unsigned int i = 0; i < dimensionOfInputPoints; ++i) {
        if(i == 0) {
            fprintf(output_file, "x = [");
        }
        else{
            fprintf(output_file, "y = [");
        }
        for(unsigned int j = 0; j < numberOfInputPoints; ++j) {
            fprintf(output_file, "%lf, ", h_pointCloud[i*numberOfInputPoints + j]);
        }
        fprintf(output_file, "]\n");
    }
    fclose(output_file);
}

void printKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, DIVISION_METHOD divMethod, unsigned int leafSize, KDTree tree, H2Opus_Real* d_pointCloud) {
    int *h_segmentIndices = (int*)malloc(numberOfInputPoints*sizeof(int));
    cudaMemcpy(h_segmentIndices, tree.segmentIndices, numberOfInputPoints*sizeof(int), cudaMemcpyDeviceToHost);

    int maxNumSegments;
    if(divMethod == FULL_TREE) {
        maxNumSegments = 1<<(getMaxSegmentSize(numberOfInputPoints, leafSize).second);
    }
    else {
        maxNumSegments = (numberOfInputPoints + leafSize - 1)/leafSize;
    }   
    int *h_segmentOffsets = (int*)malloc((maxNumSegments + 1)*sizeof(int));
    cudaMemcpy(h_segmentOffsets, tree.segmentOffsets, (maxNumSegments + 1)*sizeof(int), cudaMemcpyDeviceToHost);

    char filename[100] = "results/kdtree.txt";
    FILE *output_file = fopen(filename, "w");
    fprintf(output_file, "segmentIndices\n");
    for(unsigned int i = 0; i < numberOfInputPoints; ++i) {
        fprintf(output_file, "%d ", h_segmentIndices[i]);
    }
    fprintf(output_file, "\n\n");
    fprintf(output_file, "segmentOffsets\n");
    for(unsigned int i = 0; i < (maxNumSegments + 1); ++i) {
        fprintf(output_file, "%d ", h_segmentOffsets[i]);
    }
    fprintf(output_file, "\n\n\n");

    H2Opus_Real *h_pointCloud = (H2Opus_Real*)malloc(numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real));
    cudaMemcpy(h_pointCloud, d_pointCloud, numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    fprintf(output_file, "n = %d\n", numberOfInputPoints);
    fprintf(output_file, "bs = %d\n", leafSize);
    for(unsigned int i = 0; i < dimensionOfInputPoints; ++i) {
        if(i == 0) {
            fprintf(output_file, "X = [");
        }
        else{
            fprintf(output_file, "Y = [");
        }

        for(int j = 0; j < numberOfInputPoints/leafSize; ++j) {
            fprintf(output_file, "[ ");
            for(unsigned int k = 0; k < leafSize; ++k) {
                fprintf(output_file, "%lf, ", h_pointCloud[i*numberOfInputPoints + h_segmentIndices[j*leafSize + k]]);
            }
            fprintf(output_file, "], ");
        }
        fprintf(output_file, "]\n");
    }

    fclose(output_file);
}