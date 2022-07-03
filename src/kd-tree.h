#ifndef __H2OPUS_KD_TREE_H__
#define __H2OPUS_KD_TREE_H__

#include <math.h>

template <class T> class H2OpusDataSet
{
  public:
    virtual int getDimension() const = 0;
    virtual size_t getDataSetSize() const = 0;
    virtual T getDataPoint(size_t index, int dimension_index) const = 0;
    virtual ~H2OpusDataSet(){};
};

#endif
