#ifndef TLR_Matrix_H
#define TLR_Matrix_H

struct TLR_Matrix{
    unsigned int n;
    unsigned int blockSize;
    unsigned int maxRank;
    unsigned int nBlocks;
    unsigned int * blockRanks;
    unsigned int * lowRankBlockPointers;
    float * u;
    float * v;
    float * diagonal;

    void createRandomMatrix(unsigned int xn, unsigned int xblockSize);
    void createOutputMatrix(unsigned int xn, unsigned int xblockSize);
    void del();
};


#endif