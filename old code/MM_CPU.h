float[] CPU_MM(TLR_Matrix matrix1, TLR_Matrix matrix2, defaultN, defaultBlockSize){
    int matrix1_c[defaultN*defaultN];
    int matrix2_c[defaultN*defaultN];

    unsigned int nBlocks = (defaultN + numThreadsPerBlock.x -1)/numThreadsPerBlock.x;

    for(unsigned int i=0; i<nBlocks; ++i){
        for(unsigned int j=0; j<defaultBlockSize;++j){
            for(int k=0; k<defaultBlockSize; ++k){
                matrix1_c[i*defaultN*defaultBlockSize + i*defaultBlockSize + j*defaultN + k] = matrix1.diagonal[i*defaultBlockSize + j*defaultN + k];
                matrix2_c[i*defaultN*defaultBlockSize + i*defaultBlockSize + j*defaultN + k] = matrix2.diagonal[i*defaultBlockSize + j*defaultN + k];
            }
        }
    }

    float* result1 = (float*)malloc(defaultBlockSize*defaultBlockSize);
    float* result2 = (float*)malloc(defaultBlockSize*defaultBlockSize);
    for(unsigned int i=0; i < nBlocks*nBlocks; ++i){
        if(i%(nBlocks+1) == 0){
            continue;
        }
        result1 = tt_cpu(&matrix1.u[matrix1.lowRankBlockPointers[i]*sizeof(float)], &matrix1.v[matrix1.lowRankBlockPointers[i]*sizeof(float)], result, defaultBlockSize, defaultBlockSize,matrix1.blockRanks[i]);
        result2 = tt_cpu(&matrix2.u[matrix2.lowRankBlockPointers[i]*sizeof(float)], &matrix2.v[matrix2.lowRankBlockPointers[i]*sizeof(float)], result, defaultBlockSize, defaultBlockSize,matrix2.blockRanks[i]);

        for(unsigned int j=0; j<defaultBlockSize;++j){
            for(int k=0; k<defaultBlockSize; ++k){
                matrix1_c[i*defaultN + j*defaultN + i*defaultBlockSize + k] = result1[j*defaultBlockSize + k];
                matrix2_c[i*defaultN + j*defaultN + i*defaultBlockSize + k]] = result2[j*defaultBlockSize + k];
            }
        }
    }
}

void tt_cpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for(unsigned int i = 0; i < K; ++i) {
                sum += A[row*K + i]*B[i*N + col];
            }
            C[row*N + col] = sum;
        }
    }
}