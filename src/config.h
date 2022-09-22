#ifndef CONFIG_H
#define CONFIG_H

#include <unistd.h>
#include <stdio.h>
#include <cstring>
#include <cstdlib>

enum DIVISION_METHOD
{
    POWER_OF_TWO_ON_LEFT,
    DIVIDE_IN_HALF,
    FULL_TREE,
};

static DIVISION_METHOD parseDivMethod(const char *s)
{
    if (strcmp(s, "poftwo") == 0)
    {
        return POWER_OF_TWO_ON_LEFT;
    }
    else if (strcmp(s, "divhalf") == 0)
    {
        return DIVIDE_IN_HALF;
    }
    else if(strcmp(s, "fulltree") == 0){
        return FULL_TREE;
    }
    else
    {
        fprintf(stderr, "Unrecognized -i option: %s\n", s);
        exit(0);
    }
}

static const char *asString(DIVISION_METHOD div_method)
{
    switch (div_method)
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

struct Config
{
    // Version version;
    DIVISION_METHOD div_method;
    unsigned int n; // TODO: rename
    unsigned int dim; // TODO: rename
    unsigned int bucket_size;
    float tol; // TODO: rename to something more representative (e.g., lowestLevelTolerance)
};

static Config parseArgs(int argc, char **argv)
{
    Config config;
    config.div_method = POWER_OF_TWO_ON_LEFT;
    config.dim = 2;
    config.n = 128;
    config.bucket_size = 32;
    config.tol = 1e-5;

    int opt;
    while ((opt = getopt(argc, argv, "m:d:n:b:t")) >= 0)
    {
        switch (opt)
        {
        case 'n':
            config.n = atoi(optarg);
            break;
        case 'b':
            config.bucket_size = atoi(optarg);
            break;
        case 'e':
            config.tol = atof(optarg);
            break;
        case 'm':
            config.div_method = parseDivMethod(optarg);
            break;
        case 'd':
            config.dim = atoi(optarg);
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            // TODO: print list of options and what they mean
            exit(0);
        }
    }

    return config;
}

#endif

