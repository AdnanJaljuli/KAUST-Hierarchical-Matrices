__global__ void fillArrays2(int batchCount, int max_rows, int max_cols, int* d_rows_batch, int* d_cols_batch, H2Opus_Real** d_M_ptrs, H2Opus_Real* d_M, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch){
    for(unsigned int i=0; i<batchCount; ++i){
        d_rows_batch[i] = max_rows;
        d_cols_batch[i] = max_cols;
        d_ldm_batch[i] = max_rows;
        d_lda_batch[i] = max_rows;
        d_ldb_batch[i] = max_rows;
        d_M_ptrs[i] = d_M + i*max_rows*max_cols;
    }
}

__global__ void printOutput(H2Opus_Real** d_A_ptrs, H2Opus_Real** d_B_ptrs, int* k, int batchCount){
    printf("ks\n");
    for(unsigned int i=0;i<batchCount; ++i){
        printf("%d ", k[i]);
    }
}