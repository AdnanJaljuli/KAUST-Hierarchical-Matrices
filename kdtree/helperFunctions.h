#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#define numTimers 9

void printCountersInFile(unsigned int iteration, unsigned int num_segments, float* times){
    char filename[100] = "results/times.csv";

    FILE *output_file = fopen(filename, "a");
    fprintf(output_file, "%u,", iteration);
    fprintf(output_file, "%u", num_segments);

    for(unsigned int i = 0;i<numTimers; ++i){
        fprintf(output_file,",%f",times[i]);
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

int getMaxSegmentSize(int n, int bucket_size){
    while(n > bucket_size){
        n = (n + 1)/2;
    }
    return n;
}

#endif