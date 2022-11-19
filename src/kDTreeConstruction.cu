
#include "kDTreeConstruction.cuh"
#include "kDTreeHelpers.cuh"
#include <cub/cub.cuh>

// TODO: clean this file
void constructKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int bucketSize, KDTree &kDTree, H2Opus_Real* d_pointCloud) {

    int *d_dimxNSegmentOffsets;
    H2Opus_Real *d_kDTreePoints;
    H2Opus_Real *d_kDTreePointsOutput;
    int  *d_indexMapOutput;
    H2Opus_Real *d_reduceIn;
    H2Opus_Real *d_minSegmentItem;
    H2Opus_Real *d_maxSegmentItem;
    H2Opus_Real *d_segmentSpan;
    int* d_segmentSpanOffsets;
    cub::KeyValuePair<int, H2Opus_Real> *d_segmentSpanReduction;

    cudaMalloc((void**) &d_dimxNSegmentOffsets, (kDTree.numSegments*dimensionOfInputPoints + 1)*sizeof(int));
    cudaMalloc((void**) &d_kDTreePoints, numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_kDTreePointsOutput, numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_indexMapOutput, numberOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_reduceIn, numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_minSegmentItem, (kDTree.numSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_maxSegmentItem, (kDTree.numSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_segmentSpan, (kDTree.numSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_segmentSpanOffsets, (kDTree.numSegments + 1)*sizeof(int));
    cudaMalloc((void**) &d_segmentSpanReduction, (kDTree.numSegments + 1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));

    unsigned int currentNumSegments = 1;
    unsigned int numSegmentsReduce = currentNumSegments*dimensionOfInputPoints;
    unsigned int currentSegmentSize = upperPowerOfTwo(numberOfInputPoints);
    void *d_tempStorage;
    size_t tempStorageBytes;
    int *d_temp;

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
    initIndexMap <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, kDTree);

    while(currentSegmentSize > bucketSize) {

        numThreadsPerBlock = 1024;
        numBlocks = (currentNumSegments + 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillOffsets <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, currentSegmentSize, kDTree, d_dimxNSegmentOffsets);

        numThreadsPerBlock = 1024;
        numBlocks = (numberOfInputPoints*dimensionOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillReductionArray <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, d_pointCloud, kDTree.segmentIndices, d_reduceIn);
        cudaDeviceSynchronize();

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedReduce::Min(d_tempStorage, tempStorageBytes, d_reduceIn, d_minSegmentItem,
            numSegmentsReduce, d_dimxNSegmentOffsets, d_dimxNSegmentOffsets + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedReduce::Min(d_tempStorage, tempStorageBytes, d_reduceIn, d_minSegmentItem,
            numSegmentsReduce, d_dimxNSegmentOffsets, d_dimxNSegmentOffsets + 1);
        cudaFree(d_tempStorage);

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedReduce::Max(d_tempStorage, tempStorageBytes, d_reduceIn, d_maxSegmentItem,
            numSegmentsReduce, d_dimxNSegmentOffsets, d_dimxNSegmentOffsets + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedReduce::Max(d_tempStorage, tempStorageBytes, d_reduceIn, d_maxSegmentItem,
            numSegmentsReduce, d_dimxNSegmentOffsets, d_dimxNSegmentOffsets + 1);
        cudaFree(d_tempStorage);

        // TODO: launch more threads
        numThreadsPerBlock = 1024;
        numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
        findSpan <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, d_minSegmentItem, d_maxSegmentItem, d_segmentSpan, d_segmentSpanOffsets);
        cudaDeviceSynchronize();

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedReduce::ArgMax(d_tempStorage, tempStorageBytes, d_segmentSpan, d_segmentSpanReduction,
            currentNumSegments, d_segmentSpanOffsets, d_segmentSpanOffsets + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedReduce::ArgMax(d_tempStorage, tempStorageBytes, d_segmentSpan, d_segmentSpanReduction,
            currentNumSegments, d_segmentSpanOffsets, d_segmentSpanOffsets + 1);
        cudaFree(d_tempStorage);

        numThreadsPerBlock = 1024;
        numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillKeysIn <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, currentSegmentSize, d_kDTreePoints, d_segmentSpanReduction, kDTree.segmentIndices, d_pointCloud);
        cudaDeviceSynchronize();

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
            d_kDTreePoints, d_kDTreePointsOutput, kDTree.segmentIndices, d_indexMapOutput,
            numberOfInputPoints, currentNumSegments, kDTree.segmentOffsets, kDTree.segmentOffsets + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
            d_kDTreePoints, d_kDTreePointsOutput, kDTree.segmentIndices, d_indexMapOutput,
            numberOfInputPoints, currentNumSegments, kDTree.segmentOffsets, kDTree.segmentOffsets + 1);
        cudaFree(d_tempStorage);

        d_temp = kDTree.segmentIndices;
        kDTree.segmentIndices = d_indexMapOutput;
        d_indexMapOutput = d_temp;

        currentSegmentSize >>= 1;
        currentNumSegments = (numberOfInputPoints + currentSegmentSize - 1)/currentSegmentSize;
        numSegmentsReduce = currentNumSegments*dimensionOfInputPoints;
    }
    fillOffsets <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, currentSegmentSize, kDTree, d_dimxNSegmentOffsets);

    cudaFree(d_dimxNSegmentOffsets);
    cudaFree(d_kDTreePoints);
    cudaFree(d_kDTreePointsOutput);
    cudaFree(d_indexMapOutput);
    cudaFree(d_reduceIn);
    cudaFree(d_minSegmentItem);
    cudaFree(d_maxSegmentItem);
    cudaFree(d_segmentSpan);
    cudaFree(d_segmentSpanOffsets);
    cudaFree(d_segmentSpanReduction);
}