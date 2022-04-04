#ifndef __TLR_EXAMPLE_H__
#define __TLR_EXAMPLE_H__

#include "kd-tree.h"
#include <vector>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h> 

// This is an example of a custom data set class
template <class T> class PointCloud : public H2OpusDataSet<T>
{
  public:
    int dimension;
    size_t num_points;
    std::vector<std::vector<T>> pts;

    PointCloud(int dim, size_t num_pts)
    {
        this->dimension = dim;
        this->num_points = num_pts;

        pts.resize(dim);
        for (int i = 0; i < dim; i++){
            pts[i].resize(num_points);
        }
    }

    int getDimension() const
    {
        return dimension;
    }

    size_t getDataSetSize() const
    {
        return num_points;
    }

    inline T getDataPoint(size_t idx, int dim) const
    {
        return pts[dim][idx];
    }

    inline void getDataPoint(size_t idx, T *out) const
    {
        for (int i = 0; i < dimension; i++)
            out[i] = pts[i][idx];
    }
};

template <typename T>
void generateGrid(PointCloud<T> &pt_cloud)
{
    srand (time(NULL));
    int randomNum;
    for (int i = 0; i < pt_cloud.num_points; ++i)
    {
        for(int j=0; j<pt_cloud.dimension; ++j){
            randomNum = rand() % 100000 + 1;
            pt_cloud.pts[j][i] = randomNum;
        }
    }
}

#endif