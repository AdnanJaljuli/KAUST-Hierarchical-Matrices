#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <typeinfo>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
using namespace std;

int main(int argc, char *argv[]){
    double myText;

    // Read from the text file
    ifstream MyReadFile("batchedMatrix.txt");

    
    while (getline (MyReadFile, myText)) {
        // Output the text from the file
        cout << myText;
    }

    // Close the file
    MyReadFile.close();
}