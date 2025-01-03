
#include "helperFunctions.cuh"
#include "HMatrix.cuh"
#include "kDTree.cuh"
#include "kDTreeHelpers.cuh"
#include <curand.h>

void convertColumnMajorToMorton(unsigned int numSegments, unsigned int maxSegmentSize, uint64_t rankSum, TLR_Matrix matrix, TLR_Matrix &mortonMatrix) {

    cudaMalloc((void**) &mortonMatrix.U, rankSum*maxSegmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.V, rankSum*maxSegmentSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &mortonMatrix.blockOffsets, numSegments*numSegments*sizeof(int));
    cudaMalloc((void**) &mortonMatrix.blockRanks, numSegments*numSegments*sizeof(int));
    cudaMalloc((void**) &mortonMatrix.diagonal, static_cast<uint64_t>(numSegments)*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numSegments*numSegments + 1024 - 1)/1024;
    copyCMRanksToMORanks <<< numBlocks, numThreadsPerBlock >>> (numSegments, maxSegmentSize, matrix.blockRanks, mortonMatrix.blockRanks);

    // scan mortonMatrix ranks
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, numSegments*numSegments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, mortonMatrix.blockRanks, mortonMatrix.blockOffsets, numSegments*numSegments);
    cudaFree(d_temp_storage);

    int* h_matrix_offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    int* h_mortonMatrix_offsets = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(h_matrix_offsets, matrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mortonMatrix_offsets, mortonMatrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    int* h_matrix_ranks = (int*)malloc(numSegments*numSegments*sizeof(int));
    cudaMemcpy(h_matrix_ranks, matrix.blockRanks, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i=0; i<numSegments*numSegments; ++i){
        int MOIndex = CMIndextoMOIndex(numSegments, i);
        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (h_matrix_ranks[i]*maxSegmentSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
        assert(h_matrix_ranks[i] >= 0);
        if(h_matrix_ranks[i] > 0){
            cudaMemcpy(&mortonMatrix.U[static_cast<uint64_t>(h_mortonMatrix_offsets[MOIndex])*maxSegmentSize], &matrix.U[static_cast<uint64_t>(h_matrix_offsets[i])*maxSegmentSize], static_cast<uint64_t>(h_matrix_ranks[i])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&mortonMatrix.V[static_cast<uint64_t>(h_mortonMatrix_offsets[MOIndex])*maxSegmentSize], &matrix.V[static_cast<uint64_t>(h_matrix_offsets[i])*maxSegmentSize], static_cast<uint64_t>(h_matrix_ranks[i])*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        }
    }

    cudaMemcpy(mortonMatrix.diagonal, matrix.diagonal, numSegments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void copyCMRanksToMORanks(int num_segments, int maxSegmentSize, int* matrixRanks, int* mortonMatrixRanks){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<num_segments*num_segments){
        int MOIndex = CMIndextoMOIndex(num_segments, i);
        mortonMatrixRanks[MOIndex] = matrixRanks[i];
    }
}

void generateRandomVector(unsigned int vectorWidth, unsigned int vectorHeight, H2Opus_Real *vector) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniformDouble(gen, vector, vectorWidth*vectorHeight);
    curandDestroyGenerator(gen);
}

void generateMaxRanks(unsigned int numLevels, unsigned int bucketSize, unsigned int *maxRanks) {
    for(unsigned int i = 0; i < numLevels - 2; ++i) {
        maxRanks[i] = bucketSize*(1 << i);
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
    for(unsigned int i = 0; i < HMatrixStruct.numLevels - 1; ++i) {
        unsigned int numTiles = 1<<(i + 1);
        fprintf(output_file, "[ ");
        for(unsigned int j = 0; j < HMatrixStruct.numTiles[i]; ++j) {
            uint32_t x, y;
            mortonToCM((uint32_t)HMatrixStruct.tileIndices[i][j], x, y);
            fprintf(output_file, "%d, ", x*numTiles + y);
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

void printKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, DIVISION_METHOD divMethod, unsigned int bucketSize, KDTree tree, H2Opus_Real* d_pointCloud) {
    int *h_segmentIndices = (int*)malloc(numberOfInputPoints*sizeof(int));
    cudaMemcpy(h_segmentIndices, tree.segmentIndices, numberOfInputPoints*sizeof(int), cudaMemcpyDeviceToHost);

    int maxNumSegments;
    if(divMethod == FULL_TREE) {
        maxNumSegments = 1<<(getMaxSegmentSize(numberOfInputPoints, bucketSize).second);
    }
    else {
        maxNumSegments = (numberOfInputPoints + bucketSize - 1)/bucketSize;
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
    fprintf(output_file, "bs = %d\n", bucketSize);
    for(unsigned int i = 0; i < dimensionOfInputPoints; ++i) {
        if(i == 0) {
            fprintf(output_file, "X = [");
        }
        else{
            fprintf(output_file, "Y = [");
        }

        for(int j = 0; j < numberOfInputPoints/bucketSize; ++j) {
            fprintf(output_file, "[ ");
            for(unsigned int k = 0; k < bucketSize; ++k) {
                fprintf(output_file, "%lf, ", h_pointCloud[i*numberOfInputPoints + h_segmentIndices[j*bucketSize + k]]);
            }
            fprintf(output_file, "], ");
        }
        fprintf(output_file, "]\n");
    }

    fclose(output_file);
}