#include "siftdectector.h"
#include <iostream>
SIFTDectector::SIFTDectector()
{

}

bool SIFTDectector::DescriptorOnCustomPoints(const cv::Mat &img, const std::vector<bool> &visibles, const Eigen::Matrix2Xf &points_pos, const Eigen::VectorXf &scales, Eigen::VectorXf &descriptors, const Eigen::VectorXf &oritations, bool compute_ori)
{
    if(img.type()!=CV_32F)
    {
        std::cout<<"SIFTDectector: detector img has to be CV_32F type!"<<std::endl;
        return false;
    }
    if(!img.isContinuous())
    {
        std::cout<<"SIFTDectector: detector img has to be continuous!"<<std::endl;
        return false;
    }
    float *data = (float *)img.data;
    float *ikeys = new float[5*points_pos.cols()];
    copy_data_to_frames(points_pos, scales, oritations, ikeys);
    if(!check_sorted(scales.data(), scales.rows()))
    {
        qsort (ikeys, points_pos.cols(), 5 * sizeof(float), korder) ;
    }

    VlSiftFilt        *filt ;
    filt = vl_sift_new (img.cols, img.rows, -1, 3, 0) ;
    bool first = true ;
    int nkeys = points_pos.cols();
    int i = 0;
    descriptors.resize(nkeys*128);
    float *result_data=descriptors.data();
    while (i<nkeys)
    {
        int err ;
        /* Calculate the GSS for the next octave .................... */
        if (first)
        {
            err = vl_sift_process_first_octave (filt, data) ;
            first = false ;
        }
        else
        {
            err = vl_sift_process_next_octave  (filt) ;
        }
        if (err) break ;
        /* For each keypoint ........................................ */
        for ( ; i < nkeys ; ++i)
        {
            //process un visible point
            if(!visibles[int(ikeys[5*i+4])])
            {
                vl_sift_pix rbuf [128] ;
                for(int tnum=0; tnum<128; tnum++)
                    rbuf[tnum] = 0.0;
                float *save_data = result_data + int(ikeys[5*i+4])*128;
                memcpy(save_data, rbuf, 128*sizeof(float));
                continue;
            }

            double                angles [4] ;
            int                   nangles ;
            VlSiftKeypoint        ik ;
            VlSiftKeypoint const *k ;

            /* Obtain keypoint orientations ........................... */
            vl_sift_keypoint_init (filt, &ik,
                                 ikeys [5 * i + 0] ,
                                 ikeys [5 * i + 1] ,
                                 ikeys [5 * i + 2]) ;
            if (ik.o != vl_sift_get_octave_index (filt))
                break ;
            k = &ik ;
            /* optionally compute orientations too */
            if (compute_ori)
            {
                nangles = vl_sift_calc_keypoint_orientations(filt, angles, k) ;
            }
            else
            {
                angles [0] = ikeys [5 * i + 3] ;
                nangles    = 1 ;
            }
            //use only one oritation
            if(nangles==0)
            {
               // delete [] ikeys;
               // return false;
			   angles [0] = 0.0;
			   nangles = 1;
            }
            vl_sift_pix rbuf [128] ;
            /* compute descriptor */
            vl_sift_calc_keypoint_descriptor (filt, rbuf, k, angles[0]) ;
            float *save_data = result_data + int(ikeys[5*i+4])*128;
            memcpy(save_data, rbuf, 128*sizeof(float));
        }
    }
    if(i<nkeys)
    {
        delete [] ikeys;
        return false;
    }
    delete [] ikeys;
    vl_sift_delete(filt);
    return true;
}

bool SIFTDectector::check_sorted(const float *scales, int size)
{
    for(int i=0; i<size-1; i++)
    {
        if(scales[i]>scales[i+1]+1.0e-10)
            return false;
    }
    return true;
}
