#ifndef FEATUREDETECTOR_H
#define FEATUREDETECTOR_H
#include <opencv/cv.h>
#include <Eigen/Dense>

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Ordering of tuples by increasing scale
 **
 ** @param a tuple.
 ** @param b tuple.
 **
 ** @return @c a[2] < b[2]
 **/

static int
korder (void const* a, void const* b) {
  float x = ((float*) a) [2] - ((float*) b) [2] ;
  if (x < 0) return -1 ;
  if (x > 0) return +1 ;
  return 0 ;
}

class FeatureDetector
{
public:
    FeatureDetector();
    virtual bool DescriptorOnCustomPoints(const cv::Mat &img, const Eigen::Matrix2Xf &points_pos, const Eigen::VectorXf &scales, Eigen::VectorXf &descriptors,const Eigen::VectorXf &oritations=Eigen::VectorXf(), bool compute_ori=true)=0;
protected:
    //img data need single and row-major 0~1 and with correct size
    //everytime with copy, is not efficient!
    bool convert_Mat2floatdata(const cv::Mat &img, float* data);
    //Eigen cols have to be the same, frames have to be set the correct size, frame has to add one index to save sort order
    void copy_data_to_frames(const Eigen::Matrix2Xf &postions, const Eigen::VectorXf &scales, const Eigen::VectorXf &oritations, float *frames);
};

#endif // FEATUREDETECTOR_H
