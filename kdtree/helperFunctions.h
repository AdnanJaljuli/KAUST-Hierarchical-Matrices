#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <utility>
#include <cstdint> 
#define numTimers 10

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

void printSigmas(double* S, uint64_t num_segments, uint64_t maxSegmentSize, int bucket_size, int n, int segment){
    FILE *fp;
    fp = fopen("results/sigma_values.csv", "a");// "w" means that we are going to write on this file
    if(segment == 0){
        fprintf(fp, "bucket size: %d, n: %d, num segments: %d,\n", bucket_size, n, num_segments);
    }

    for(unsigned int i=0; i<num_segments; ++i){
        for(unsigned int j=0; j<maxSegmentSize; ++j){
            fprintf(fp, "%lf, ", S[i*maxSegmentSize + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp); //Don't forget to close the file when finished
}

void printKs(unsigned int* K, uint64_t num_segments, uint64_t maxSegmentSize, int bucket_size, int n, int segment, float eps){
    FILE *fp;
    fp = fopen("results/K_values.csv", "a");// "w" means that we are going to write on this file
    if(segment == 0){
        fprintf(fp, "bucket size: %d, n: %d, num segments: %d, epsilon: %f\n", bucket_size, n, num_segments, eps);
    }

    for(unsigned int i=0; i<num_segments; ++i){
            fprintf(fp, "%u, ", K[i]);
    }
    fprintf(fp, "\n");
    fclose(fp); //Don't forget to close the file when finished
}

#endif