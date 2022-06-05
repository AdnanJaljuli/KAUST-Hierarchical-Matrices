#include "TLR_Matrix.h"

TLR_Matrix AllocateInputMatrix(TLR_Matrix CPU_matrix){
    TLR_Matrix GPU_matrix;
    GPU_matrix.n = CPU_matrix.n;
    GPU_matrix.maxRank = CPU_matrix.maxRank;
    GPU_matrix.blockSize = CPU_matrix.blockSize;
    GPU_matrix.nBlocks = CPU_matrix.nBlocks;
    GPU_matrix.blockRanks = CPU_matrix.blockRanks;
    GPU_matrix.lowRankBlockPointers = CPU_matrix.lowRankBlockPointers;
    GPU_matrix.u = CPU_matrix.u;
    GPU_matrix.v = CPU_matrix.v;
    GPU_matrix.diagonal = CPU_matrix.diagonal;

    unsigned int * blockRanks_d;
    unsigned int * lowRankBlockPointers_d;
    float * u_d;
    float * v_d;
    float * diagonal_d;

    unsigned int numberOfElements = 0;
    for (unsigned int i = 0; i < CPU_matrix.nBlocks*CPU_matrix.nBlocks-CPU_matrix.nBlocks; ++i){
        numberOfElements += (CPU_matrix.blockSize*CPU_matrix.blockRanks[i]);
    }

    cudaMalloc((void**) &blockRanks_d, sizeof(unsigned int)*(CPU_matrix.nBlocks*CPU_matrix.nBlocks - CPU_matrix.nBlocks));
    cudaMalloc((void**) &lowRankBlockPointers_d, sizeof(unsigned int)*(CPU_matrix.nBlocks*CPU_matrix.nBlocks - CPU_matrix.nBlocks));
    cudaMalloc((void**) &u_d, sizeof(float)*numberOfElements);
    cudaMalloc((void**) &v_d, sizeof(float)*numberOfElements);
    cudaMalloc((void**) &diagonal_d, sizeof(float)*CPU_matrix.nBlocks*CPU_matrix.blockSize*CPU_matrix.blockSize);

    cudaMemcpy(blockRanks_d, GPU_matrix.blockRanks, sizeof(unsigned int)*((CPU_matrix.nBlocks * CPU_matrix.nBlocks)-CPU_matrix.nBlocks), cudaMemcpyHostToDevice);
    cudaMemcpy(lowRankBlockPointers_d, GPU_matrix.lowRankBlockPointers, sizeof(unsigned int)*((CPU_matrix.nBlocks * CPU_matrix.nBlocks)-CPU_matrix.nBlocks), cudaMemcpyHostToDevice);
    cudaMemcpy(u_d, GPU_matrix.u, sizeof(float)*numberOfElements, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, GPU_matrix.v, sizeof(float)*numberOfElements, cudaMemcpyHostToDevice);
    cudaMemcpy(diagonal_d, GPU_matrix.diagonal, sizeof(float)*CPU_matrix.nBlocks * CPU_matrix.blockSize * CPU_matrix.blockSize, cudaMemcpyHostToDevice);

    return GPU_matrix;
}

TLR_Matrix AllocateOutputMatrix(TLR_Matrix CPU_matrix){
    TLR_Matrix GPU_matrix;
    GPU_matrix.n = CPU_matrix.n;
    GPU_matrix.maxRank = CPU_matrix.maxRank;
    GPU_matrix.blockSize = CPU_matrix.blockSize;
    GPU_matrix.nBlocks = CPU_matrix.nBlocks;

    unsigned int numberOfElements = GPU_matrix.n*GPU_matrix.n;

    float * matrix_d;
    cudaMalloc((void**) &matrix_d, sizeof(float)*numberOfElements);

    return GPU_matrix;
}

void cudaFreeMatrix(TLR_Matrix matrix){
    cudaFree(matrix.blockRanks);
    cudaFree(matrix.lowRankBlockPointers);
    cudaFree(matrix.u);
    cudaFree(matrix.v);
    cudaFree(matrix.diagonal);
}