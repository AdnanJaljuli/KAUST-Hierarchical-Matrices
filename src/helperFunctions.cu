
#include "helperFunctions.cuh"
#include "HMatrix.cuh"
#include "kDTree.cuh"
#include "kDTreeHelpers.cuh"
#include <curand.h>

__global__ void copyCMRanksToMORanks(int numLeaves, int tileSize, int* tileScanRanks, int* mortonTileRanks){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numLeaves*numLeaves) {
        int MOIndex = columnMajor2Morton(numLeaves, i);
        int prevScanRanks = (i == 0) ? 0 : tileScanRanks[i - 1];
        int rank = tileScanRanks[i] - prevScanRanks;
        mortonTileRanks[MOIndex] = rank;
    }
}

template <class T>
void convertColumnMajorToMorton(TLR_Matrix matrix, TLR_Matrix *mortonMatrix) {

    mortonMatrix->tileSize = matrix.tileSize;
    mortonMatrix->numTilesInAxis = matrix.numTilesInAxis;
    mortonMatrix->rankSum = matrix.rankSum;

    mortonMatrix->d_U.resize(matrix.rankSum*matrix.tileSize);
    mortonMatrix->d_V.resize(matrix.rankSum*matrix.tileSize);
    cudaMalloc((void**) &mortonMatrix->d_tileOffsets, matrix.numTilesInAxis*matrix.numTilesInAxis*sizeof(int));

    int *d_tileRanks;
    cudaMalloc((void**) &d_tileRanks, matrix.numTilesInAxis*matrix.numTilesInAxis*sizeof(int));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (matrix.numTilesInAxis*matrix.numTilesInAxis + numThreadsPerBlock - 1)/numThreadsPerBlock;
    copyCMRanksToMORanks <<< numBlocks, numThreadsPerBlock >>> (
        matrix.numTilesInAxis,
        matrix.tileSize,
        matrix.d_tileOffsets,
        d_tileRanks);

    // scan mortonMatrix ranks
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_tileRanks, mortonMatrix->d_tileOffsets, matrix.numTilesInAxis*matrix.numTilesInAxis);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_tileRanks, mortonMatrix->d_tileOffsets, matrix.numTilesInAxis*matrix.numTilesInAxis);
    cudaFree(d_temp_storage);

    cudaFree(d_tileRanks);

    int* h_matrix_offsets = (int*)malloc(matrix.numTilesInAxis*matrix.numTilesInAxis*sizeof(int));
    int* h_mortonMatrix_offsets = (int*)malloc(matrix.numTilesInAxis*matrix.numTilesInAxis*sizeof(int));
    cudaMemcpy(h_matrix_offsets, matrix.d_tileOffsets, matrix.numTilesInAxis*matrix.numTilesInAxis*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mortonMatrix_offsets, mortonMatrix->d_tileOffsets, matrix.numTilesInAxis*matrix.numTilesInAxis*sizeof(int), cudaMemcpyDeviceToHost);

    T *d_UPtr_CM = thrust::raw_pointer_cast(matrix.d_U.data());
    T *d_VPtr_CM = thrust::raw_pointer_cast(matrix.d_V.data());

    T *d_UPtr_MO = thrust::raw_pointer_cast(mortonMatrix->d_U.data());
    T *d_VPtr_MO = thrust::raw_pointer_cast(mortonMatrix->d_V.data());

    for(unsigned int i = 0; i < matrix.numTilesInAxis*matrix.numTilesInAxis; ++i){
        int MOIndex = columnMajor2Morton(matrix.numTilesInAxis, i);
        int CMScanRank = (i == 0) ? 0 : h_matrix_offsets[i - 1];
        int mortonScanRank = (MOIndex == 0) ? 0 : h_mortonMatrix_offsets[MOIndex - 1];
        int rank = h_mortonMatrix_offsets[MOIndex] - mortonScanRank;

        assert(rank >= 0);
        if(rank > 0){
            cudaMemcpy(&d_UPtr_MO[mortonScanRank*matrix.tileSize], &d_UPtr_CM[CMScanRank*matrix.tileSize], rank*matrix.tileSize*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&d_VPtr_MO[mortonScanRank*matrix.tileSize], &d_VPtr_CM[CMScanRank*matrix.tileSize], rank*matrix.tileSize*sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    gpuErrchk(cudaPeekAtLastError());
}

template void convertColumnMajorToMorton <H2Opus_Real> (TLR_Matrix matrix, TLR_Matrix *mortonMatrix);

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
            morton2columnMajor((uint32_t)(HMatrixStruct.tileIndices[i][j]), x, y);
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
    int *h_leafIndices = (int*)malloc(numberOfInputPoints*sizeof(int));
    cudaMemcpy(h_leafIndices, tree.leafIndices, numberOfInputPoints*sizeof(int), cudaMemcpyDeviceToHost);

    int maxNumSegments;
    if(divMethod == FULL_TREE) {
        maxNumSegments = 1<<(getMaxSegmentSize(numberOfInputPoints, leafSize).second);
    }
    else {
        maxNumSegments = (numberOfInputPoints + leafSize - 1)/leafSize;
    }   
    int *h_leafOffsets = (int*)malloc((maxNumSegments + 1)*sizeof(int));
    cudaMemcpy(h_leafOffsets, tree.leafOffsets, (maxNumSegments + 1)*sizeof(int), cudaMemcpyDeviceToHost);

    char filename[100] = "results/kdtree.txt";
    FILE *output_file = fopen(filename, "w");
    fprintf(output_file, "leafIndices\n");
    for(unsigned int i = 0; i < numberOfInputPoints; ++i) {
        fprintf(output_file, "%d ", h_leafIndices[i]);
    }
    fprintf(output_file, "\n\n");
    fprintf(output_file, "leafOffsets\n");
    for(unsigned int i = 0; i < (maxNumSegments + 1); ++i) {
        fprintf(output_file, "%d ", h_leafOffsets[i]);
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
                fprintf(output_file, "%lf, ", h_pointCloud[i*numberOfInputPoints + h_leafIndices[j*leafSize + k]]);
            }
            fprintf(output_file, "], ");
        }
        fprintf(output_file, "]\n");
    }

    fclose(output_file);
}