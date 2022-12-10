#include <stdint.h>
#include <stdio.h>

__device__ uint32_t morton1(uint32_t x)
{
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

__device__ void mortonToCM(uint32_t d, uint32_t &x, uint32_t &y) {
    x = morton1(d);
    y = morton1(d >> 1);
}


__global__ void testFunction() {
    uint32_t d, x, y;
    for(int i = 0; i < 16; ++i) {
        d = i;
        mortonToCM(d, x, y);
        printf("d: %d   %d   %d\n", d, x, y);
    }
}

int main() {
    testFunction<<<1, 1 >>> ();
    cudaDeviceSynchronize();
}