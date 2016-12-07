#include "featuredetector.h"
#include <iostream>
FeatureDetector::FeatureDetector()
{

}

bool FeatureDetector::convert_Mat2floatdata(const cv::Mat &img, float *data)
{
    cv::Mat float_img;
    if(img.channels()==3)
    {
        cv::Mat temp;
        cv::cvtColor(img,temp,CV_BGR2GRAY);
        temp.convertTo(float_img, CV_32F,1.0);

    }
    else if(img.channels()==1)
    {
        if(img.type()==CV_8U)
        {
            img.convertTo(float_img, CV_32F, 1.0);
        }
        else
        {
            std::cout<<"FeatureDectetor: img has 1 channel, but is not a gray img, can not convert to single data!"<<std::endl;
            return false;
        }
    }
    else
    {
        std::cout<<"FeatureDectetor: img has not 3 or 1 channels, can not convert to single data!"<<std::endl;
        return false;
    }
    if(!float_img.isContinuous())
    {
        std::cout<<"FeatureDectetor: float img is not continuous!"<<std::endl;
        return false;
    }
    float *now_data = data;
    for(int i=0; i<float_img.rows; i++)
    {
        const float *ptr = float_img.ptr<float>(i);
        memcpy(now_data, ptr, sizeof(float)*float_img.cols);
        now_data+=float_img.cols;
    }
    return true;
}

void FeatureDetector::copy_data_to_frames(const Eigen::Matrix2Xf &postions, const Eigen::VectorXf &scales, const Eigen::VectorXf &oritations, float *frames)
{
    const float *data_p = postions.data();
    const float *data_s = scales.data();
    bool use_oritation = true;
    if(oritations.size()==0)
        use_oritation = false;
    const float *data_o = oritations.data();
    float *now = frames;
    for(int i=0;i<postions.cols();i++)
    {
        memcpy(now, data_p, sizeof(float)*2);
        now+=2;
        data_p+=2;
        *now = *data_s;
        now++;  data_s++;
        if(use_oritation)
        {
            *now = *data_o;
            now++;  data_o++;
        }
        else
        {
            *now = 0.0;
            now++;
        }
        *now = float(i);
        now++;
    }
}
