#include "TLR_Matrix.h"

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <time.h>
using namespace std;

void TLR_Matrix::del(){
    free(blockRanks);
    free(blockOffsets);
    free(U);
    free(V);
    free(diagonal);

}

