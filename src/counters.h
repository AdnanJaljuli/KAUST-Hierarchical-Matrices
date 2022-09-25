
#ifndef COUNTERS_H
#define COUNTERS_H

#include <string.h>
#include "config.h"

enum counterName { TOTAL_TIME = 0, NUM_COUNTERS };

struct Counters {
    cudaEvent_t startEvent[NUM_COUNTERS];
    cudaEvent_t endEvent[NUM_COUNTERS];
    float totalTime[NUM_COUNTERS];
};

static void initCounters(Counters* counters) {
    #if USE_COUNTERS
    for(unsigned int i = 0; i < NUM_COUNTERS; ++i) {
        counters->totalTime[i] = 0;
    }
    #endif
}

static void startTime(counterName counter, Counters* counters) {
    #if USE_COUNTERS
    cudaEventCreate(&counters->startEvent[counter]);
    cudaEventRecord(counters->startEvent[counter]);
    #endif
}

static void endTime(counterName counter, Counters *counters) {
    #if USE_COUNTERS
    // counters->totalTime[counetr] += clock64() - counters->startTime[counter];
    cudaEventCreate(&counters->endEvent[counter]);
    cudaEventRecord(counters->endEvent[counter]);
    cudaEventSynchronize(counters->endEvent[counter]);
    float time;
    cudaEventElapsedTime(&counters->totalTime[counter], counters->startEvent[counter], counters->endEvent[counter]);
    #endif
}

static void printCountersInFile(Config config, Counters *counters) {
    char filename[100] = "results/times.csv";
    FILE *output_file = fopen(filename, "a");
    // NUMBER_OF_INPUT_POINTS = 0, DIMENSTION_OF_INPUT_POINTS, BUCKET_SIZE

    for(unsigned int i = 0; i < NUM_COUNTERS; ++i){
        fprintf(output_file,"%f, ", counters->totalTime[i]);
        printf("counters total time: %f\n", counters->totalTime[i]);
    }

    fprintf(output_file, "\n");
    fclose(output_file);
}


#endif