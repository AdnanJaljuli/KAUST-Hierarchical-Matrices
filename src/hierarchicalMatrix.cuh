
#ifndef __HIERARCHICALMATRIX_H__
#define __HIERARCHICALMATRIX_H__

#include "config.h"
#include "counters.h"
#include "hierarchicalMatrixHelpers.cuh"
#include "kDTree.h"
#include "TLRMatrix.h"

void genereateHierarchicalMatrix(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R) {

    const int numHMatrixLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    printf("numHMatrixLevels: %d\n", numHMatrixLevels);
    // TODO: make this into a struct
    int** HMatrixExistingRanks = (int**)malloc((numHMatrixLevels - 1)*sizeof(int*));
    int** HMatrixExistingTiles = (int**)malloc((numHMatrixLevels - 1)*sizeof(int*));
    int numExistingTiles = numSegments*(numSegments - 1);
    cudaMalloc((void**) &HMatrixExistingRanks[numHMatrixLevels - 2], numExistingTiles*sizeof(int));
    cudaMalloc((void**) &HMatrixExistingTiles[numHMatrixLevels - 2], numExistingTiles*sizeof(int));

    int *d_rowsBatch, *d_colsBatch, *d_ranks;
    int *d_LDABatch, *d_LDBBatch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_APtrs, **d_BPtrs;

    // unsigned int numThreadsPerBlock = 1024;
    // unsigned int numBlocks = (numSegments*numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    // fillFirstLevel <<< numBlocks, numThreadsPerBlock >>> (numSegments, HMatrixExistingTiles[numHMatrixLevels - 2], HMatrixExistingRanks[numHMatrixLevels - 2], mortonOrderedMatrix.blockRanks);
    fillInitialHMatrixLevel(numSegments, HMatrixExistingTiles[numHMatrixLevels - 2], HMatrixExistingRanks[numHMatrixLevels - 2], mortonOrderedMatrix.blockRanks);
    return;

    int maxRows = segmentSize;
    int maxCols = segmentSize;
    int maxRank = maxCols; // this is taken as is from the kblas_ARA repo

    unsigned int tileSize = bucketSize;
    bool stop = false;

    // TODO: fix the number of iterations.
    for(unsigned int level = numHMatrixLevels - 2; level > 0; --level) {
        // TODO: set cudaMalloc and cudaFrees to outside the loop
        int* d_num_ops;
        cudaMalloc((void**) &d_num_ops, sizeof(int));
        int num_ops;
        cudaMemset(d_num_ops, 0, sizeof(int));
        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (numExistingTiles + numThreadsPerBlock - 1)/numThreadsPerBlock;
        // TODO: instead of using atmoicAdds, let each thread write to a bit vector and then do a reduce
        calcNumOps<<<numBlocks, numThreadsPerBlock>>> (numExistingTiles, d_num_ops, HMatrixExistingTiles[level - 1]);        
        cudaMemcpy(&num_ops, d_num_ops, sizeof(int), cudaMemcpyDeviceToHost);
        printf("level: %d   num ops: %d\n", level, num_ops);
        cudaFree(d_num_ops);

        int* d_activeTiles;
        int* d_activeRanks;
        cudaMalloc((void**) &d_activeTiles, 4*num_ops*sizeof(int));
        cudaMalloc((void**) &d_activeRanks, 4*num_ops*sizeof(int));

        // TODO: parallelize
        fillActiveTiles<<<1, 1>>>(numExistingTiles, d_activeTiles, HMatrixExistingTiles[level - 1], d_activeRanks, HMatrixExistingRanks[level - 1]);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        printK<<<1, 1>>>(d_activeTiles, num_ops*4);

        maxRows <<= 1;
        maxCols <<= 1;
        maxRank <<= 1;
        // tolerance *= 2;
        printf("max rows: %d\n", maxRows);
        printf("tolerance: %f\n", tolerance);

        // TODO: find a tight upper limit and malloc and free before and after the loop
        gpuErrchk(cudaMalloc((void**) &d_ranks, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_rowsBatch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_colsBatch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_LDABatch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_LDBBatch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_APtrs, num_ops*sizeof(H2Opus_Real*)));
        gpuErrchk(cudaMalloc((void**) &d_BPtrs, num_ops*sizeof(H2Opus_Real*)));
        gpuErrchk(cudaMalloc((void**) &d_A, num_ops*maxRows*maxRank*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_B, num_ops*maxRows*maxRank*sizeof(H2Opus_Real)));

        numThreadsPerBlock = 1024;
        numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillLRARAArrays<<<numBlocks, numThreadsPerBlock>>>(num_ops, maxRows, maxCols, d_rowsBatch, d_colsBatch, d_LDABatch, d_LDBBatch);

        generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, maxRows*maxCols, num_ops, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, maxRows*maxCols, num_ops, 0);
        gpuErrchk(cudaPeekAtLastError());

        kblasHandle_t kblasHandle;
        kblasRandState_t randState;
        kblasCreate(&kblasHandle);

        kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
        gpuErrchk(cudaPeekAtLastError());

        kblasEnableMagma(kblasHandle);
        kblas_gesvj_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, maxCols, num_ops);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, bucketSize, num_ops);
        kblasAllocateWorkspace(kblasHandle);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        // TODO: write a unit test for the lr_kblas_ara_batch function. 
        // TODO: optimize maxCols. maxCols shouldn't be equal to maxRows, instead, its values should depend on the ranks of the tiles
        int lr_ARA_return = lr_kblas_ara_batch(kblasHandle, d_rowsBatch, d_colsBatch, mortonOrderedMatrix.U, mortonOrderedMatrix.V, d_activeRanks, mortonOrderedMatrix.blockOffsets, d_activeTiles,
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks,
            tolerance, maxRows, maxCols, maxRank, 32, ARA_R, randState, 0, num_ops
        );
        cudaDeviceSynchronize();
        assert(lr_ARA_return == 1);
        gpuErrchk(cudaPeekAtLastError());

        // TODO: move this error checking to its own function
        #if EXPAND_MATRIX
        cudaDeviceSynchronize();
        H2Opus_Real* expandedHMatrix;
        cudaMalloc((void**) &expandedHMatrix, num_ops*maxRows*maxCols*sizeof(H2Opus_Real));
        dim3 hm_numBlocks(2, 2*num_ops);
        dim3 hm_numThreadsPerBlock(32, 32);
        expandHMatrixLevel<<<hm_numBlocks, hm_numThreadsPerBlock>>>(num_ops, 64, 64, d_A, d_B, d_ranks, expandedHMatrix);

        cudaMemset(d_error, 0, sizeof(H2Opus_Real));
        cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));
        errorInHMatrix<<<hm_numBlocks, hm_numThreadsPerBlock>>>(numSegments, maxSegmentSize, num_ops, maxRows, maxCols, expandedHMatrix, d_denseMatrix, d_activeTiles, d_error, d_tmp);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        printf("h matrix error: %lf\n", sqrt(h_error)/sqrt(h_tmp));
        cudaFree(expandedHMatrix);
        #endif
        break;

        // TODO: optimize the bit vector: use an array of longs instead.
        int* d_old_bit_vector;
        int* d_new_bit_vector;
        int* d_old_bit_vector_scan;
        int* d_new_bit_vector_scan;
        gpuErrchk(cudaMalloc((void**) &d_old_bit_vector, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_new_bit_vector, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_old_bit_vector_scan, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_new_bit_vector_scan, num_ops*sizeof(int)));

        numThreadsPerBlock = 1024;
        numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_ops, tileSize, d_ranks, d_activeRanks, d_new_bit_vector, d_old_bit_vector);
        cudaDeviceSynchronize();
        tileSize <<= 1;

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_old_bit_vector, d_old_bit_vector_scan, num_ops);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_old_bit_vector, d_old_bit_vector_scan, num_ops);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_new_bit_vector, d_new_bit_vector_scan, num_ops);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_new_bit_vector, d_new_bit_vector_scan, num_ops);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* d_newLevelCount;
        int* newLevelCount = (int*)malloc(sizeof(int));
        cudaMalloc((void**) &d_newLevelCount, sizeof(int));
        getNewLevelCount<<<1, 1>>>(num_ops, d_new_bit_vector, d_new_bit_vector_scan, d_newLevelCount);
        cudaMemcpy(newLevelCount, d_newLevelCount, sizeof(int), cudaMemcpyDeviceToHost);
        numExistingTiles = *newLevelCount;
        printf("new level count %d\n", numExistingTiles);

        if(*newLevelCount == 0) {
            stop = true;
        }
        else {
            gpuErrchk(cudaMalloc((void**) &HMatrixExistingRanks[level - 1], *newLevelCount*sizeof(int)));
            gpuErrchk(cudaMalloc((void**) &HMatrixExistingTiles[level - 1], *newLevelCount*sizeof(int)));

            numThreadsPerBlock = 1024;
            numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
            fillNewLevel<<<numBlocks, numThreadsPerBlock>>>(num_ops, d_new_bit_vector, d_new_bit_vector_scan, d_ranks, HMatrixExistingRanks[level - 1], d_activeTiles, HMatrixExistingTiles[level - 1]);
            copyTilesToNewLevel<<<numBlocks, numThreadsPerBlock>>>(num_ops, d_new_bit_vector, mortonOrderedMatrix, d_A, d_B, d_ranks, d_activeTiles, maxRows, maxCols);
            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            // TODO: clean previous ranks and active tiles arrays
        }
        kblasDestroy(&kblasHandle);
        kblasDestroyRandState(randState);
        free(newLevelCount);
        cudaFree(d_newLevelCount);

        cudaFree(d_ranks);
        cudaFree(d_rowsBatch);
        cudaFree(d_colsBatch);
        cudaFree(d_LDABatch);
        cudaFree(d_LDBBatch);
        cudaFree(d_APtrs);
        cudaFree(d_BPtrs);
        cudaFree(d_A);
        cudaFree(d_B);
        if(stop){
            break;
        }
        break;
    }
}
#endif