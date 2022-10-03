#ifndef HIERARCHICALMATRIX_FUNCTIONS_H
#define HIERARCHICALMATRIX_FUNCTIONS_H

static __global__ void getNewLevelCount(int num_ops, int* d_new_bit_vector, int* d_new_bit_vector_scan, int* d_newLevelCount){
    d_newLevelCount[0] = d_new_bit_vector_scan[num_ops - 1] + d_new_bit_vector[num_ops - 1];
}

static __global__ void copyTilesToNewLevel(int num_ops, int* bit_vector, TLR_Matrix mortonMatrix, H2Opus_Real* d_A, H2Opus_Real* d_B, int* new_ranks, int* old_active_tiles, int row, int col){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_ops){
        if(bit_vector[i] == 1){
            // TODO: fix the address to where the function will copy
            // TODO: use multiple streams
            cudaMemcpyAsync(&mortonMatrix.U[mortonMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_A[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(&mortonMatrix.V[mortonMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_B[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
        }
    }
}

static __global__ void calcNumOps(int num_existing_tiles, int* num_ops, int* availableTiles){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_existing_tiles){
        if(availableTiles[i]%4 == 0){
            bool flag = true;
            for(int j=1; j<4; ++j){
                if(availableTiles[i+j] != availableTiles[i]+j){
                    flag = false;
                    break;
                }
            }
            if(flag){
                atomicAdd(num_ops, 1);
            }
        }
    }
}

static __global__ void expandHMatrixLevel(int num_ops, int max_rows, int max_cols, H2Opus_Real* d_A, H2Opus_Real* d_B, int* d_ranks, H2Opus_Real* expandedMatrix){
    int col = threadIdx.x + blockIdx.x*(max_cols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(max_rows/2);
    int block = blockIdx.y/2;
    H2Opus_Real sum = 0;

    for(unsigned int i=0; i<d_ranks[block]; ++i){
        sum += d_A[max_rows*max_cols*block + i*max_rows + row]*d_B[max_rows*max_cols*block + i*max_rows + col];
    }
    expandedMatrix[block*max_rows*max_cols + col*max_rows + row] = sum;
}

static __global__ void errorInHMatrix(int num_segments, int maxSegmentSize, int num_ops, int max_rows, int max_cols, H2Opus_Real* expandedMatrix, H2Opus_Real* d_denseMatrix, int* activeTiles, H2Opus_Real* d_error, H2Opus_Real* d_tmp){
    int col = threadIdx.x + blockIdx.x*(max_cols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(max_rows/2);
    int block = blockIdx.y/2;

    int MOIndex = activeTiles[block*4]/4;
    int i = morton1(MOIndex);
    int j = morton1(MOIndex >> 1);

    H2Opus_Real x = d_denseMatrix[(col + i*max_cols)*num_segments*maxSegmentSize + j*max_rows + row];
    H2Opus_Real y = expandedMatrix[block*max_rows*max_cols + col*max_rows + row];
    atomicAdd(d_tmp, x*x);
    atomicAdd(d_error, (x-y)*(x-y));
}

void genereateHierarchicalMatrix(int n, int bucket_size, int numSegments, int maxSegmentSize, int numLevels, TLR_Matrix mortonMatrix, int** HMatrixExistingRanks, int** HMatrixExistingTiles){
    int numExistingTiles = numSegments*(numSegments-1);

    int *d_rows_batch, *d_cols_batch, *d_ranks;
    int *d_lda_batch, *d_ldb_batch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_A_ptrs, **d_B_ptrs;

    cudaMalloc((void**) &HMatrixExistingRanks[numLevels - 2], numExistingTiles*sizeof(int));
    cudaMalloc((void**) &HMatrixExistingTiles[numLevels - 2], numExistingTiles*sizeof(int));

    // TODO: parallelize
    fillFirstLevelExistingArrays<<<1, 1>>>(numSegments, HMatrixExistingTiles[numLevels - 2], HMatrixExistingRanks[numLevels - 2], mortonMatrix.blockRanks);
    unsigned int tile_size = bucket_size;
    bool stop = false;

    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));

    #if 0
    // TODO: fix the number of iterations.
    for(unsigned int level = numLevels - 1; level > 0; --level){
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

        max_rows <<= 1;
        max_cols <<= 1;
        max_rank <<= 1;
        // tolerance *= 2;
        printf("max rows: %d\n", max_rows);
        printf("tolerance: %f\n", tolerance);

        // TODO: find a tight upper limit and malloc and free before and after the loop
        gpuErrchk(cudaMalloc((void**) &d_ranks, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_rows_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_cols_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_lda_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_ldb_batch, num_ops*sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_A_ptrs, num_ops*sizeof(H2Opus_Real*)));
        gpuErrchk(cudaMalloc((void**) &d_B_ptrs, num_ops*sizeof(H2Opus_Real*)));
        gpuErrchk(cudaMalloc((void**) &d_A, num_ops*max_rows*max_rank*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_B, num_ops*max_rows*max_rank*sizeof(H2Opus_Real)));

        numThreadsPerBlock = 1024;
        numBlocks = (num_ops + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillLRARAArrays<<<numBlocks, numThreadsPerBlock>>>(num_ops, max_rows, max_cols, d_rows_batch, d_cols_batch, d_lda_batch, d_ldb_batch);

        generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols, num_ops, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols, num_ops, 0);
        gpuErrchk(cudaPeekAtLastError());

        kblasHandle_t kblas_handle_2;
        kblasRandState_t rand_state_2;
        kblasCreate(&kblas_handle_2);

        kblasInitRandState(kblas_handle_2, &rand_state_2, 1<<15, 0);
        gpuErrchk(cudaPeekAtLastError());

        kblasEnableMagma(kblas_handle_2);
        kblas_gesvj_batch_wsquery<H2Opus_Real>(kblas_handle_2, max_rows, max_cols, num_ops);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle_2, bucket_size, num_ops);
        kblasAllocateWorkspace(kblas_handle_2);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        // TODO: write a unit test for the lr_kblas_ara_batch function. 
        // TODO: optimize max_cols. max_cols shouldn't be equal to max_rows, instead, its values should depend on the ranks of the tiles
        int lr_ARA_return = lr_kblas_ara_batch(kblas_handle_2, d_rows_batch, d_cols_batch, mortonMatrix.U, mortonMatrix.V, d_activeRanks, mortonMatrix.blockOffsets, d_activeTiles,
            d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks,
            tolerance, max_rows, max_cols, max_rank, 32, ARA_R, rand_state_2, 0, num_ops
        );
        cudaDeviceSynchronize();
        assert(lr_ARA_return == 1);
        gpuErrchk(cudaPeekAtLastError());

        // TODO: move this error checking to its own function
        #if EXPAND_MATRIX
        cudaDeviceSynchronize();
        H2Opus_Real* expandedHMatrix;
        cudaMalloc((void**) &expandedHMatrix, num_ops*max_rows*max_cols*sizeof(H2Opus_Real));
        dim3 hm_numBlocks(2, 2*num_ops);
        dim3 hm_numThreadsPerBlock(32, 32);
        expandHMatrixLevel<<<hm_numBlocks, hm_numThreadsPerBlock>>>(num_ops, 64, 64, d_A, d_B, d_ranks, expandedHMatrix);

        cudaMemset(d_error, 0, sizeof(H2Opus_Real));
        cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));
        errorInHMatrix<<<hm_numBlocks, hm_numThreadsPerBlock>>>(numSegments, maxSegmentSize, num_ops, max_rows, max_cols, expandedHMatrix, d_denseMatrix, d_activeTiles, d_error, d_tmp);
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
        fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_ops, tile_size, d_ranks, d_activeRanks, d_new_bit_vector, d_old_bit_vector);
        cudaDeviceSynchronize();
        tile_size <<= 1;

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
            copyTilesToNewLevel<<<numBlocks, numThreadsPerBlock>>>(num_ops, d_new_bit_vector, mortonMatrix, d_A, d_B, d_ranks, d_activeTiles, max_rows, max_cols);
            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            // TODO: clean previous ranks and active tiles arrays
        }
        kblasDestroy(&kblas_handle_2);
        kblasDestroyRandState(rand_state_2);
        free(newLevelCount);
        cudaFree(d_newLevelCount);

        cudaFree(d_ranks);
        cudaFree(d_rows_batch);
        cudaFree(d_cols_batch);
        cudaFree(d_lda_batch);
        cudaFree(d_ldb_batch);
        cudaFree(d_A_ptrs);
        cudaFree(d_B_ptrs);
        cudaFree(d_A);
        cudaFree(d_B);
        if(stop){
            break;
        }
        break;
    }
    #endif
}
#endif