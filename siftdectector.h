#ifndef SIFTDECTECTOR_H
#define SIFTDECTECTOR_H
#include "featuredetector.h"
#include "sift.h"
class SIFTDectector: public FeatureDetector
{
public:
    SIFTDectector();
    //scale need to sort from min to max, compute_ori=true, oritations will be recomputed, false, direct use oritations
    //as possible as to do not copy data, make efficient
    //output descriptors are unit vector
    bool DescriptorOnCustomPoints(const cv::Mat &img, const std::vector<bool> &visibles,const Eigen::Matrix2Xf &points_pos, const Eigen::VectorXf &scales, Eigen::VectorXf &descriptors,const Eigen::VectorXf &oritations=Eigen::VectorXf(), bool compute_ori=true);
private:
    bool check_sorted(const float *scales, int size);
};

#endif // SIFTDECTECTOR_H
