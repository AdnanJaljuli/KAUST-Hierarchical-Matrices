#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <cub/cub.cuh>
#include <assert.h>

#include "cublas_v2.h"

#include "kblas.h"
#include "batch_rand.h"
#include "batch_pstrf.h"
#include "batch_block_copy.h"
#include "batch_ara.h"
#include "magma_auxiliary.h"

__global__ void lr_sampling_batch(
    H2Opus_Real* U_batch, H2Opus_Real* V_batch, int* ranks, int* scan_ranks,
    H2Opus_Real** B_batch, H2Opus_Real** A_batch, int batch_count, int transpose) {
        __shared__ H2Opus_Real U_s[32*32];
        __shared__ H2Opus_Real V_s[32*32];
        __shared__ H2Opus_Real Omega_s[32*32];

        if(transpose == 0){
            for(unsigned int i=0; i<2; ++i){
                if(threadIdx.x < ranks[i*batch_count + blockIdx.x]){
                    U_s[threadIdx.x*32 + threadIdx.y] = U_batch[scan_ranks[i*batch_count + blockIdx.x]*32 + threadIdx.x*32 + threadIdx.y];
                    V_s[threadIdx.x*32 + threadIdx.y] = V_batch[scan_ranks[i*batch_count + blockIdx.x]*32 + threadIdx.x*32 + threadIdx.y];
                }
                Omega_s[threadIdx.x*32 + threadIdx.y] = B_batch[blockIdx.x][threadIdx.x*32*2 + i*32 + threadIdx.y];
                __syncthreads();

                H2Opus_Real accum = 0;

                for(unsigned int j=0; j<32; ++j){
                    H2Opus_Real uv = 0;
                    for(unsigned int k=0; k<ranks[i*batch_count + blockIdx.x]; ++k){
                        uv += U_s[k*32 + threadIdx.y]*V_s[k*32 + threadIdx.x];
                    }
                    accum += uv*Omega_s[threadIdx.x*32 + threadIdx.y];
                }
                A_batch[blockIdx.x][threadIdx.x*32*2 + i*32 + threadIdx.y] += accum;
                __syncthreads();
            }           
        }
        else {
            for(unsigned int i=0; i<2; ++i){
                if(threadIdx.x < ranks[(blockIdx.x%2)*batch_count + blockIdx.x - (blockIdx.x%2) + i]){
                    U_s[threadIdx.x*32 + threadIdx.y] = U_batch[scan_ranks[(blockIdx.x%2)*batch_count + blockIdx.x - (blockIdx.x%2) + i]*32 + threadIdx.x*32 + threadIdx.y];
                    V_s[threadIdx.x*32 + threadIdx.y] = V_batch[scan_ranks[(blockIdx.x%2)*batch_count + blockIdx.x - (blockIdx.x%2) + i]*32 + threadIdx.x*32 + threadIdx.y];
                }
                Omega_s[threadIdx.x*32 + threadIdx.y] = B_batch[blockIdx.x][threadIdx.x*32*2 + i*32 + threadIdx.y];
                __syncthreads();

                H2Opus_Real accum = 0;

                for(unsigned int j=0; j<32; ++j){
                    H2Opus_Real uv = 0;
                    for(unsigned int k=0; k<ranks[(blockIdx.x%2)*batch_count + blockIdx.x - (blockIdx.x%2) + i]; ++k){
                        uv += U_s[k*32 + threadIdx.x]*V_s[k*32 + threadIdx.y];
                    }
                    accum += uv*Omega_s[threadIdx.x*32 + threadIdx.y];
                }
                A_batch[blockIdx.x][threadIdx.x*32*2 + i*32 + threadIdx.y] += accum;
                __syncthreads();
            }           
        }
}

int sample_batch(H2Opus_Real* U_batch, H2Opus_Real* V_batch, int* ranks, int* scan_ranks, H2Opus_Real** B_batch, int* ldb_batch, int* samples_batch, H2Opus_Real** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) {
    dim3 numBlocks(2*num_ops, 1);
    dim3 numThreadsPerBlock(32, 32);
    lr_sampling_batch<<<numBlocks, numThreadsPerBlock>>>(U_batch, V_batch, ranks, scan_ranks, B_batch, A_batch, num_ops, transpose);
    cudaDeviceSynchronize();
    return 1;
}