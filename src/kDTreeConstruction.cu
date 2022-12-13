
#include "kDTreeConstruction.cuh"
#include "boundingBoxes.h"
#include "config.h"
#include "kDTreeHelpers.cuh"

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// TODO: clean this file
void constructKDTree(
    KDTree &kDTree, 
    H2Opus_Real* d_pointCloud,
    DIVISION_METHOD divMethod) {

        int maxNumSegments;
        if(divMethod == FULL_TREE) {
            maxNumSegments = 1<<(getMaxSegmentSize(kDTree.N, kDTree.leafSize).second);
        }
        else {
            maxNumSegments = (kDTree.N + kDTree.leafSize - 1)/kDTree.leafSize;
        }

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

        cudaMalloc((void**) &d_dimxNSegmentOffsets, (maxNumSegments*kDTree.nDim + 1)*sizeof(int));
        cudaMalloc((void**) &d_kDTreePoints, kDTree.N*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_kDTreePointsOutput, kDTree.N*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_indexMapOutput, kDTree.N*sizeof(int));
        cudaMalloc((void**) &d_reduceIn, kDTree.N*kDTree.nDim*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_minSegmentItem, (maxNumSegments + 1)*kDTree.nDim*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_maxSegmentItem, (maxNumSegments + 1)*kDTree.nDim*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_segmentSpan, (maxNumSegments + 1)*kDTree.nDim*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_segmentSpanOffsets, (maxNumSegments + 1)*sizeof(int));
        cudaMalloc((void**) &d_segmentSpanReduction, (maxNumSegments + 1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));

        if(divMethod == FULL_TREE) {
            cudaMalloc((void**) &d_aux_offsets_sort, (maxNumSegments + 1)*sizeof(int));
            cudaMalloc((void**) &A, (maxNumSegments + 1)*sizeof(int));
            cudaMalloc((void**) &B, kDTree.N*sizeof(int));
            cudaMalloc((void**) &d_bin_search_output, kDTree.N*sizeof(int));
            cudaMalloc((void**) &d_input_search, kDTree.N*sizeof(int));
        }

        H2Opus_Real *d_bufferBBMax, *d_bufferBBMin;
        cudaMalloc((void**) &d_bufferBBMax, (maxNumSegments + 1)*kDTree.nDim*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_bufferBBMin, (maxNumSegments + 1)*kDTree.nDim*sizeof(H2Opus_Real));

        unsigned int currentNumSegments = 1;
        unsigned int numSegmentsReduce = currentNumSegments*kDTree.nDim;
        // TODO: make largestSegmentSize and currentSegmentSize one variable instead of two
        // unsigned int currentSegmentSize = upperPowerOfTwo(kDTree.N);
        // unsigned int currentSegmentSize = kDTree.N;
        unsigned int currentSegmentSize;
        if(divMethod == POWER_OF_TWO_ON_LEFT) {
            currentSegmentSize = upperPowerOfTwo(kDTree.N);
        }
        else {
            currentSegmentSize = kDTree.N;
        }

        void *d_tempStorage;
        size_t tempStorageBytes;
        int *d_temp;

        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (kDTree.N + numThreadsPerBlock - 1)/numThreadsPerBlock;
        if(divMethod == POWER_OF_TWO_ON_LEFT) {
            initIndexMap <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree);
        }
        else {
            initIndexMap <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, kDTree, d_input_search, d_dimxNSegmentOffsets);
        }

        unsigned int level = 0;

        while(currentSegmentSize > kDTree.leafSize)
        {
            if(divMethod == POWER_OF_TWO_ON_LEFT) {
                numThreadsPerBlock = 1024;
                numBlocks = (currentNumSegments + 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillOffsets <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, currentNumSegments, currentSegmentSize, kDTree, d_dimxNSegmentOffsets);
            }

            numThreadsPerBlock = 1024;
            numBlocks = (kDTree.N*kDTree.nDim + numThreadsPerBlock - 1)/numThreadsPerBlock;
            fillReductionArray <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, d_pointCloud, kDTree.segmentIndices, d_reduceIn);

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

            // copy segmented min and max to bounding boxes
            copyMaxandMinToBoundingBoxes(
                kDTree.boundingBoxes.levels[level],
                d_maxSegmentItem,
                d_minSegmentItem,
                level,
                kDTree.nDim,
                currentNumSegments,
                d_bufferBBMax, d_bufferBBMin);

            numThreadsPerBlock = 1024;
            numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
            findSpan <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, currentNumSegments, d_minSegmentItem, d_maxSegmentItem, d_segmentSpan, d_segmentSpanOffsets);

            d_tempStorage = NULL;
            tempStorageBytes = 0;
            cub::DeviceSegmentedReduce::ArgMax(d_tempStorage, tempStorageBytes, d_segmentSpan, d_segmentSpanReduction,
                currentNumSegments, d_segmentSpanOffsets, d_segmentSpanOffsets + 1);
            cudaMalloc(&d_tempStorage, tempStorageBytes);
            cub::DeviceSegmentedReduce::ArgMax(d_tempStorage, tempStorageBytes, d_segmentSpan, d_segmentSpanReduction,
                currentNumSegments, d_segmentSpanOffsets, d_segmentSpanOffsets + 1);
            cudaFree(d_tempStorage);

            numThreadsPerBlock = 1024;
            numBlocks = (kDTree.N + numThreadsPerBlock - 1)/numThreadsPerBlock;
            if(divMethod == POWER_OF_TWO_ON_LEFT) {
                fillKeysIn <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, currentSegmentSize, d_kDTreePoints, d_segmentSpanReduction, kDTree.segmentIndices, d_pointCloud);
            }
            else {
                thrust::device_ptr<int> A = thrust::device_pointer_cast((int *)kDTree.segmentOffsets), B = thrust::device_pointer_cast((int *)d_input_search);
                thrust::device_vector<int> d_bin_search_output(kDTree.N);
                thrust::upper_bound(A, A + currentNumSegments + 1, B, B + kDTree.N, d_bin_search_output.begin(), thrust::less<int>());
                d_thrust_v_bin_search_output = thrust::raw_pointer_cast(&d_bin_search_output[0]);
                fillKeysIn <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, d_kDTreePoints, d_segmentSpanReduction, kDTree.segmentIndices, d_pointCloud, kDTree.segmentOffsets, d_thrust_v_bin_search_output);
            }

            d_tempStorage = NULL;
            tempStorageBytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
                d_kDTreePoints, d_kDTreePointsOutput, kDTree.segmentIndices, d_indexMapOutput,
                kDTree.N, currentNumSegments, kDTree.segmentOffsets, kDTree.segmentOffsets + 1);
            cudaMalloc(&d_tempStorage, tempStorageBytes);
            cub::DeviceSegmentedRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
                d_kDTreePoints, d_kDTreePointsOutput, kDTree.segmentIndices, d_indexMapOutput,
                kDTree.N, currentNumSegments, kDTree.segmentOffsets, kDTree.segmentOffsets + 1);
            cudaFree(d_tempStorage);

            d_temp = kDTree.segmentIndices;
            kDTree.segmentIndices = d_indexMapOutput;
            d_indexMapOutput = d_temp;

            if(divMethod == POWER_OF_TWO_ON_LEFT) {
                currentSegmentSize >>= 1;
                currentNumSegments = (kDTree.N + currentSegmentSize - 1)/currentSegmentSize;
            }
            else {
                numThreadsPerBlock = 1024;
                numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillOffsetsSort <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, currentNumSegments, kDTree.segmentOffsets, d_aux_offsets_sort);

                d_temp = d_aux_offsets_sort;
                d_aux_offsets_sort = kDTree.segmentOffsets;
                kDTree.segmentOffsets = d_temp;
                currentNumSegments *= 2;

                numThreadsPerBlock = 1024;
                numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillOffsetsReduce <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, currentNumSegments, kDTree.segmentOffsets, d_dimxNSegmentOffsets);

                ++currentSegmentSize;
                currentSegmentSize >>= 1;
            }

            numSegmentsReduce = currentNumSegments*kDTree.nDim;
            ++level;
        }

        if(divMethod == POWER_OF_TWO_ON_LEFT) {
            fillOffsets <<< numBlocks, numThreadsPerBlock >>> (kDTree.N, kDTree.nDim, currentNumSegments, currentSegmentSize, kDTree, d_dimxNSegmentOffsets);
        }

        kDTree.numLeaves = currentNumSegments;
        kDTree.numLevels = level + 1;

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
        cudaFree(d_bufferBBMin);
        cudaFree(d_bufferBBMax);
        if(divMethod == FULL_TREE) {
            cudaFree(d_aux_offsets_sort);
            cudaFree(A);
            cudaFree(B);
            cudaFree(d_bin_search_output);
            cudaFree(d_input_search);
        }
}