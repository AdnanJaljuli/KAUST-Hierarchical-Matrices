#include <iostream>
using namespace std;

int upper_power_of_two(int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

int main(){
    int x=7;
    int y=upper_power_of_two(x);
    cout<<y;
}