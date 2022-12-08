
#ifndef CONFIG_H
#define CONFIG_H

#include <unistd.h>
#include <stdio.h>
#include <cstring>
#include <cstdlib>

#ifndef USE_COUNTERS
#define USE_COUNTERS 0
#endif

static void usage() {
    fprintf(stderr,
            "\n"
            "Arguments :\n"
            "\n"
            "   -n <number of input points>         Used to set the number of input points.\n"
            "                                       This must be an integer.\n"
            "                                  - Default : 128\n"
            "\n"
            "   -b <bucket size>                    Used to set the bucket size, which is the smallest tile size of the TLR matrix.\n"
            "                                       This should be a power of two.\n"
            "                                  - Default : 32\n"
            "\n"
            "   -t <tolerance of lowest level>      Used to pass the tolerance for the smallest tle size.\n"
            "                                  - Default : 1e-5\n"
            "\n"
            "   -m <KD tree division method>        Used to set the method used to divide the KD tree.\n"
            "                                  - Options : powerOfTwo, divideInHald, fullTree\n"
            "                                  - Default : pooftwo\n"
            "\n"
            "   -d <dimension of input points>      Used to set the dimension of the input points. \n"
            "                                       This must be an integer.\n"
            "                                  - Default : 2\n"
            "\n"
            "   -v <width of vector>                Used to set the width of the vector that is multiplied by the hierarchical matrix\n"
            "                                  - Default: 64\n"
            "\n"
            "   -a <admissibility condition>        Used to set the admissibility condition that builds hierarchical matrix structure\n"
            "                                  - Options: weak, boxCenter\n"
            "                                  - Default: weak\n"
            "\n");
}

// TODO: remove division method
enum DIVISION_METHOD {
    POWER_OF_TWO_ON_LEFT,
    DIVIDE_IN_HALF,
    FULL_TREE,
};

static DIVISION_METHOD parseDivMethod(const char *s) {
    if (strcmp(s, "powerOfTwo") == 0)
    {
        return POWER_OF_TWO_ON_LEFT;
    }
    else if (strcmp(s, "divideInHald") == 0)
    {
        return DIVIDE_IN_HALF;
    }
    else if(strcmp(s, "fullTree") == 0){
        return FULL_TREE;
    }
    else
    {
        fprintf(stderr, "Unrecognized -i option: %s\n", s);
        exit(0);
    }
}

static const char *asString(DIVISION_METHOD divMethod) {
    switch (divMethod)
    {
        case POWER_OF_TWO_ON_LEFT:
            return "ppwer of two on left";
        case DIVIDE_IN_HALF:
            return "div in half";
        case FULL_TREE:
            return "full tree";
        default:
            fprintf(stderr, "Unrecognized instance\n");
            exit(0);
    }
}

enum ADMISSIBILITY_CONDITION {
    WEAK_ADMISSIBILITY,
    BOX_CENTER_ADMISSIBILITY,
};

static ADMISSIBILITY_CONDITION parseAdmissibilityCondition(const char *s) {
    if (strcmp(s, "weak") == 0)
    {
        return WEAK_ADMISSIBILITY;
    }
    else if (strcmp(s, "boxCenter") == 0)
    {
        return BOX_CENTER_ADMISSIBILITY;
    }
    else
    {
        fprintf(stderr, "Unrecognized -i option: %s\n", s);
        exit(0);
    }
}

struct Config {
    DIVISION_METHOD divMethod;
    ADMISSIBILITY_CONDITION admissibility;
    unsigned int numberOfInputPoints;
    unsigned int dimensionOfInputPoints;
    unsigned int bucketSize;
    unsigned int vectorWidth;
    float lowestLevelTolerance;
};

static Config parseArgs(int argc, char **argv) {
    Config config;
    config.divMethod = FULL_TREE;
    config.admissibility = WEAK_ADMISSIBILITY;
    config.numberOfInputPoints = 1024;
    config.dimensionOfInputPoints = 2;
    config.bucketSize = 32;
    config.vectorWidth = 16;
    config.lowestLevelTolerance = 1e-5;

    int opt;
    while ((opt = getopt(argc, argv, "n:b:t:m:d:v:h")) >= 0) {
        switch (opt)
        {
        case 'm':
            config.divMethod = parseDivMethod(optarg);
            break;
        case 'a':
            config.admissibility = parseAdmissibilityCondition(optarg);
            break;
        case 'n':
            config.numberOfInputPoints = atoi(optarg);
            break;
        case 'd':
            config.dimensionOfInputPoints = atoi(optarg);
            break;
        case 'b':
            config.bucketSize = atoi(optarg);
            break;
        case 'v':
            config.vectorWidth = atoi(optarg);
            break;
        case 't':
            config.lowestLevelTolerance = atof(optarg);
            break;
        case 'h':
            usage();
            exit(0);
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            exit(0);
        }
    }

    return config;
}

static void printArgs(Config config) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice name: %s\n\n", prop.name);
    printf("number of points: %d\n", config.numberOfInputPoints);
    printf("bucket size: %d\n", config.bucketSize);
    printf("lowest level tolerance: %f\n", config.lowestLevelTolerance);
    printf("dimension of input points: %d\n", config.dimensionOfInputPoints);
    printf("vector width: %u\n", config.vectorWidth);
}

#endif
