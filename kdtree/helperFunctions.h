#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <utility>
#define numTimers 11

void printCountersInFile(float* times){
    char filename[100] = "results/times.csv";

    FILE *output_file = fopen(filename, "a");
    for(unsigned int i = 0; i<numTimers; ++i){
        fprintf(output_file,"%f, ",times[i]);
    }
    fprintf(output_file, "\n");
    fclose(output_file);
}

__device__ __host__ int upper_power_of_two(int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

bool isPowerOfTwo (int x) {
    return x && (!(x&(x-1)));
}

std::pair<int, int> getMaxSegmentSize(int n, int bucket_size){
    int it=0;
    while(n > bucket_size){
        n = (n + 1)/2;
        ++it;
    }
    std::pair<int, int> p;
    p.first = n;
    p.second = it;
    return p;
}

#endif