
#include "kDTreeConstruction.cuh"
#include "config.h"
#include "kDTreeHelpers.cuh"
#include <cub/cub.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// TODO: clean this file
void constructKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, 
    unsigned int bucketSize, 
    KDTree &kDTree, 
    H2Opus_Real* d_pointCloud,
    DIVISION_METHOD divMethod) {

        int maxNumSegments;
        if(divMethod == FULL_TREE) {
            maxNumSegments = 1<<(getMaxSegmentSize(numberOfInputPoints, bucketSize).second);
        }
        else {
            maxNumSegments = (numberOfInputPoints + bucketSize - 1)/bucketSize;
        }
        printf("max num segments: %d\n", maxNumSegments);

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

        int* A;
        int* B;
        int* d_bin_search_output;
        int* d_thrust_v_bin_search_output;
        int* d_input_search;
        int* d_aux_offsets_sort;

        cudaMalloc((void**) &d_dimxNSegmentOffsets, (maxNumSegments*dimensionOfInputPoints + 1)*sizeof(int));
        cudaMalloc((void**) &d_kDTreePoints, numberOfInputPoints*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_kDTreePointsOutput, numberOfInputPoints*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_indexMapOutput, numberOfInputPoints*sizeof(int));
        cudaMalloc((void**) &d_reduceIn, numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_minSegmentItem, (maxNumSegments + 1)*dimensionOfInputPoints*sizeof(int));
        cudaMalloc((void**) &d_maxSegmentItem, (maxNumSegments + 1)*dimensionOfInputPoints*sizeof(int));
        cudaMalloc((void**) &d_segmentSpan, (maxNumSegments + 1)*dimensionOfInputPoints*sizeof(int));
        cudaMalloc((void**) &d_segmentSpanOffsets, (maxNumSegments + 1)*sizeof(int));
        cudaMalloc((void**) &d_segmentSpanReduction, (maxNumSegments + 1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));

        if(divMethod == FULL_TREE) {
            cudaMalloc((void**) &d_aux_offsets_sort, (maxNumSegments + 1)*sizeof(int));
            cudaMalloc((void**) &A, (maxNumSegments + 1)*sizeof(int));        
            cudaMalloc((void**) &B, numberOfInputPoints*sizeof(int));        
            cudaMalloc((void**) &d_bin_search_output, numberOfInputPoints*sizeof(int));        
            cudaMalloc((void**) &d_input_search, numberOfInputPoints*sizeof(int));
        }

        unsigned int currentNumSegments = 1;
        unsigned int numSegmentsReduce = currentNumSegments*dimensionOfInputPoints;
        unsigned int currentSegmentSize = upperPowerOfTwo(numberOfInputPoints);
        unsigned int largestSegmentSize = numberOfInputPoints;
        void *d_tempStorage;
        size_t tempStorageBytes;
        int *d_temp;

        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
        if(divMethod == POWER_OF_TWO_ON_LEFT) {
            initIndexMap <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, kDTree);
        }
        else {
            initIndexMap <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, kDTree, d_input_search, d_dimxNSegmentOffsets);
        }

        #if 1
        while(largestSegmentSize > bucketSize)
        #else
        while(currentSegmentSize > bucketSize)
        #endif
        {
            if(divMethod == POWER_OF_TWO_ON_LEFT) {
                numThreadsPerBlock = 1024;
                numBlocks = (currentNumSegments + 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillOffsets <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, currentSegmentSize, kDTree, d_dimxNSegmentOffsets);
            }

            numThreadsPerBlock = 1024;
            numBlocks = (numberOfInputPoints*dimensionOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
            fillReductionArray <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, d_pointCloud, kDTree.segmentIndices, d_reduceIn);

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
            if(divMethod == POWER_OF_TWO_ON_LEFT) {
                fillKeysIn <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, currentSegmentSize, d_kDTreePoints, d_segmentSpanReduction, kDTree.segmentIndices, d_pointCloud);
            }
            else {
                thrust::device_ptr<int> A = thrust::device_pointer_cast((int *)kDTree.segmentOffsets), B = thrust::device_pointer_cast((int *)d_input_search);
                thrust::device_vector<int> d_bin_search_output(numberOfInputPoints);
                thrust::upper_bound(A, A + currentNumSegments + 1, B, B + numberOfInputPoints, d_bin_search_output.begin(), thrust::less<int>());
                d_thrust_v_bin_search_output = thrust::raw_pointer_cast(&d_bin_search_output[0]);
                fillKeysIn <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, d_kDTreePoints, d_segmentSpanReduction, kDTree.segmentIndices, d_pointCloud, kDTree.segmentOffsets, d_thrust_v_bin_search_output);
            }

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

            if(divMethod == POWER_OF_TWO_ON_LEFT) {
                currentSegmentSize >>= 1;
                currentNumSegments = (numberOfInputPoints + currentSegmentSize - 1)/currentSegmentSize;
            }
            else {
                numThreadsPerBlock = 1024;
                numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillOffsetsSort <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, kDTree.segmentOffsets, d_aux_offsets_sort);

                d_temp = d_aux_offsets_sort;
                d_aux_offsets_sort = kDTree.segmentOffsets;
                kDTree.segmentOffsets = d_temp;
                currentNumSegments *= 2;

                numThreadsPerBlock = 1024;
                numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillOffsetsReduce <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, kDTree.segmentOffsets, d_dimxNSegmentOffsets);

                ++largestSegmentSize;
                largestSegmentSize >>= 1;
            }

            numSegmentsReduce = currentNumSegments*dimensionOfInputPoints;
        }

        if(divMethod == POWER_OF_TWO_ON_LEFT) {
            fillOffsets <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, currentSegmentSize, kDTree, d_dimxNSegmentOffsets);
        }

        kDTree.numSegments = currentNumSegments;

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
        if(divMethod == FULL_TREE) {
            cudaFree(d_aux_offsets_sort);
            cudaFree(A);
            cudaFree(B);
            cudaFree(d_bin_search_output);
            cudaFree(d_input_search);
        }
}