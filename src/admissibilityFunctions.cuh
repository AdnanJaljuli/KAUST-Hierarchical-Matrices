
#ifndef ADMISSIBILITY_FUNCTIONS_H
#define ADMISSIBILITY_FUNCTIONS_H

#include "precision.h"
#include "admissibilityHelpers.cuh"
#include "boundingBoxes.h"

template <class T>
class Admissibility {
    public:
        virtual bool operator()(BoundingBox node_u, BoundingBox node_v) = 0;
};

template <class T>
class BBoxCenterAdmissibility : public Admissibility<T> {
    private:
        T eta;
        unsigned int nDim;

    public:
        bool operator()(BoundingBox node_u, BoundingBox node_v) {
            T distance = BBoxCenterDistance(node_u, node_v, nDim);
            T diameter_u = BBoxDiameter(node_u, nDim);
            T diameter_v = BBoxDiameter(node_v, nDim);

            if (diameter_u == 0 || diameter_v == 0) {
                return false;
            }
            else {
                return (0.5 * (diameter_u + diameter_v) <= eta * distance);
            }
        }

        BBoxCenterAdmissibility(T eta, unsigned int nDim) {
            this->eta = eta;
            this->nDim = nDim;
        }
};

template <class T>
class WeakAdmissibility: public Admissibility<T> {
    public:
        bool operator()(BoundingBox node_u, BoundingBox node_v) {
            if(node_u.index == node_v.index) {
                return false;
            }
            else {
                return true;
            }
        }
};

#endif
