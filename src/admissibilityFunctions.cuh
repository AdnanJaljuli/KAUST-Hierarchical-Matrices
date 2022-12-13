
#ifndef ADMISSIBILITY_FUNCTIONS_H
#define ADMISSIBILITY_FUNCTIONS_H

#include "precision.h"
#include "admissibilityHelpers.cuh"
#include "boundingBoxes.h"

class Admissibility {
    public:
        virtual bool operator()(BoundingBox node_u, BoundingBox node_v) = 0;
};

class BBoxCenterAdmissibility : public Admissibility {
    private:
        H2Opus_Real eta;
        unsigned int nDim;

    public:
        bool operator()(BoundingBox node_u, BoundingBox node_v) {
            H2Opus_Real distance = BBoxCenterDistance(node_u, node_v, nDim);
            H2Opus_Real diameter_u = BBoxDiameter(node_u, nDim);
            H2Opus_Real diameter_v = BBoxDiameter(node_v, nDim);

            if (diameter_u == 0 || diameter_v == 0) {
                return false;
            }
            else {
                return (0.5 * (diameter_u + diameter_v) <= eta * distance);
            }
        }

        BBoxCenterAdmissibility(H2Opus_Real eta, unsigned int nDim) {
            this->eta = eta;
            this->nDim = nDim;
        }
};

class WeakAdmissibility: public Admissibility {
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
